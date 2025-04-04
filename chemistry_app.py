# Standard library imports
import os
import io
import json
import base64
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import (
    Draw, 
    AllChem, 
    Descriptors, 
    rdMolDescriptors
)

# Optional imports with fallbacks
try:
    import json5
    USE_JSON5 = True
except ImportError:
    st.warning("json5 not installed. For better JSON handling, run: pip install json5")
    USE_JSON5 = False

try:
    import py3Dmol
    from stmol import showmol
    _3D_VISUALIZATION_AVAILABLE = True
except ImportError:
    _3D_VISUALIZATION_AVAILABLE = False

# Check for Cairo support
try:
    from rdkit.Chem.Draw import MolDraw2DCairo
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    st.warning("Cairo support not available. For better molecular drawing, install cairo: pip install cairo")

from pydantic import BaseModel, Field

# Define a Pydantic model for chemistry questions
class ChemistryQuestion(BaseModel):
    question: str
    equations: List[str]
    solution: str
    smiles: Optional[str] = None
    requires_molecular_structure: bool

# Page configuration
st.set_page_config(
    page_title="Chemistry Assistant",
    page_icon="🧪",
    layout="wide"
)

# Configure the Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("No API key found. Please set the GEMINI_API_KEY environment variable or in Streamlit secrets.")
    st.stop()

# ------------------ UTILITY FUNCTIONS ------------------

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def fix_latex_slashes(text: str) -> str:
    """Fix LaTeX slashes by handling invalid escape sequences in JSON and restoring proper LaTeX formatting.
    
    Args:
        text: Text containing LaTeX commands
        
    Returns:
        Text with properly formatted LaTeX commands
    """
    if not text:
        return text
        
    # First, protect existing double slashes
    text = text.replace('\\\\', '§§§')
    
    # Fix invalid escape sequences that occur during JSON parsing
    latex_commands = [
        'frac', 'log', 'sin', 'cos', 'tan', 'sqrt', 'sum', 'int', 'lim',
        'rightarrow', 'leftarrow', 'leftrightarrow', 'text', 'Delta',
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
        'mu', 'pi', 'sigma', 'phi', 'omega', 'infty', 'partial', 'nabla',
        'cdot', 'times', 'div', 'approx', 'cong', 'neq', 'equiv',
        'leq', 'geq', 'pm', 'mp', 'in', 'notin', 'subset', 'supset',
        'cup', 'cap', 'mathbb', 'mathbf', 'mathrm', 'mathit', 'mathsf',
        'overrightarrow', 'underrightarrow', 'Rightarrow', 'Leftarrow',
        'longrightarrow', 'longleftarrow', 'Longrightarrow', 'Longleftarrow',
        'overline', 'underline', 'overbrace', 'underbrace', 'hat', 'bar',
        'vec', 'dot', 'ddot', 'widehat', 'widetilde', 'quad', 'qquad',
        'degree', 'circ', 'prime', 'angle', 'triangle', 'square', 'forall',
        'exists', 'nexists', 'mathcal', 'mathscr', 'limits'
    ]
    
    for cmd in latex_commands:
        # Replace single backslashes or forward slashes with double backslashes
        text = re.sub(fr'(?<!\\)(\\|/){cmd}', f'\\\\{cmd}', text)
    
    # Handle special cases for common equation symbols
    text = re.sub(r'(?<![\\a-zA-Z0-9])(->|→)', r'\\rightarrow', text)
    text = re.sub(r'(?<![\\a-zA-Z0-9])(<->|↔)', r'\\leftrightarrow', text)
    text = re.sub(r'(?<![\\a-zA-Z0-9])(=>|⇒)', r'\\Rightarrow', text)
    text = re.sub(r'(?<![\\a-zA-Z0-9])(<=|⇐)', r'\\Leftarrow', text)
    
    # Fix subscripts and superscripts (common in chemistry)
    # Ensure single-character sub/superscripts are properly formatted
    text = re.sub(r'_([a-zA-Z0-9])', r'_{\\1}', text)
    text = re.sub(r'\^([a-zA-Z0-9])', r'^{\\1}', text)
    
    # Fix common chemistry notations
    text = re.sub(r'(?<![\\a-zA-Z0-9])(H\+|H\^+)', r'H^{+}', text)
    text = re.sub(r'(?<![\\a-zA-Z0-9])(OH\-|OH\^-)', r'OH^{-}', text)
    
    # Restore protected double slashes
    text = text.replace('§§§', '\\\\')
    
    return text

def check_if_metal_atoms(mol: Chem.Mol) -> bool:
    """Check if the molecule contains metal atoms that might cause UFF/MMFF issues.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        True if molecule contains metal atoms, False otherwise
    """
    metal_symbols = {
        "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", 
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Ac", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu"
    }
    
    return any(atom.GetSymbol() in metal_symbols for atom in mol.GetAtoms())

# ------------------ GEMINI API INTERACTION ------------------

def get_gemini_response(
    prompt: str,
    response_schema: Optional[Dict] = None
) -> Union[str, Dict, None]:
    """Get a response from the Gemini API with optional structured output.
    
    Args:
        prompt: The text prompt to send to Gemini
        response_schema: Optional JSON schema for structured response
        
    Returns:
        The text response or structured data if schema was provided, or None if error occurs
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
        
        # Set up request with or without schema
        if response_schema:
            # Add explicit instructions to keep responses concise and properly formatted
            enhanced_prompt = (
                prompt + "\n\n"
                "IMPORTANT FORMATTING INSTRUCTIONS:\n"
                "1. Your response must be a single, valid JSON object\n"
                "2. Do not include any text before or after the JSON\n"
                "3. Keep all text fields under 500 characters\n"
                "4. Use proper JSON escaping for special characters\n"
                "5. Do not use comments or markdown formatting\n"
                "6. Ensure all required fields are present\n"
                "7. Use double quotes for all strings, not single quotes\n"
                "8. Do not use trailing commas\n"
                "9. Do not use line breaks within string values\n"
                "10. IMPORTANT: Your response must be complete and properly terminated\n"
                "11. Do not cut off or truncate the response\n"
                "12. Ensure all arrays and objects are properly closed\n"
                "13. If you need to continue, use '...' to indicate continuation\n"
                "14. The response must be a valid JSON object that can be parsed"
            )
            
            # Configure structured response generation
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more predictable outputs
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,  # Increased max tokens to prevent truncation
                "response_schema": response_schema,
                "response_mime_type": "application/json"
            }
            
            # Generate the response
            response = model.generate_content(
                enhanced_prompt,
                generation_config=generation_config
            )
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Try to find JSON content in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                
                # Try multiple parsing methods
                parsed_response = None
                
                # Method 1: Direct JSON parsing
                try:
                    parsed_response = json.loads(json_text)
                except json.JSONDecodeError:
                    pass
                
                # Method 2: Try json5 if available
                if not parsed_response and USE_JSON5:
                    try:
                        parsed_response = json5.loads(json_text)
                    except Exception:
                        pass
                
                # Method 3: Clean and try again
                if not parsed_response:
                    cleaned_text = json_text
                    try:
                        parsed_response = json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        pass
                
                # Method 4: Try to extract valid JSON
                if not parsed_response:
                    extracted_text = json_text
                    if extracted_text:
                        try:
                            parsed_response = json.loads(extracted_text)
                        except json.JSONDecodeError:
                            if USE_JSON5:
                                try:
                                    parsed_response = json5.loads(extracted_text)
                                except Exception:
                                    pass
                
                if parsed_response:
                    # Validate against schema
                    if response_schema and "required" in response_schema:
                        missing_fields = []
                        for field in response_schema.get("required", []):
                            if field not in parsed_response:
                                missing_fields.append(field)
                        
                        if missing_fields:
                            st.warning(f"Missing required fields: {', '.join(missing_fields)}")
                            # Try to reconstruct missing fields
                            for field in missing_fields:
                                if field in response_text:
                                    # Try to extract field value from raw text
                                    pattern = fr'"{field}".*?:.*?[",\[\{{'']([^",\}}]*)'
                                    matches = re.search(pattern, response_text)
                                    if matches:
                                        field_value = matches.group(1).strip()
                                        if field_value:
                                            parsed_response[field] = field_value
                                            st.info(f"Reconstructed field '{field}' from response text")
                    
                    # Special handling for questions array
                    if "questions" in parsed_response:
                        if not isinstance(parsed_response["questions"], list):
                            # Try to extract questions from the response text
                            questions_pattern = r'"questions"\s*:\s*\[(.*?)\]'
                            questions_match = re.search(questions_pattern, response_text, re.DOTALL)
                            if questions_match:
                                questions_text = questions_match.group(1)
                                # Try to parse individual questions
                                question_objects = []
                                question_pattern = r'\{[^}]+\}'
                                for match in re.finditer(question_pattern, questions_text):
                                    try:
                                        question_obj = json.loads(match.group(0))
                                        if isinstance(question_obj, dict):
                                            question_objects.append(question_obj)
                                    except json.JSONDecodeError:
                                        continue
                                if question_objects:
                                    parsed_response["questions"] = question_objects
                                    st.info("Reconstructed questions array from response text")
                    
                    return parsed_response
                else:
                    st.error("Failed to parse response as JSON")
                    st.code(f"Raw response:\n{response_text[:1000]}...")
                    
                    # Try to reconstruct a minimal valid response
                    if response_schema and "required" in response_schema:
                        minimal_response = {}
                        for field in response_schema.get("required", []):
                            # Try to find field values in the response text
                            pattern = fr'"{field}".*?:.*?[",\[\{{'']([^",\}}]*)'
                            matches = re.search(pattern, response_text)
                            if matches:
                                field_value = matches.group(1).strip()
                                if field_value:
                                    minimal_response[field] = field_value
                            else:
                                # Use defaults based on schema type
                                if "properties" in response_schema:
                                    field_schema = response_schema["properties"].get(field, {})
                                    field_type = field_schema.get("type", "string")
                                    
                                    if field_type == "string":
                                        minimal_response[field] = f"Unknown {field}"
                                    elif field_type == "array":
                                        minimal_response[field] = []
                                    elif field_type == "object":
                                        minimal_response[field] = {}
                                    elif field_type == "number":
                                        minimal_response[field] = 0
                        
                        if minimal_response:
                            st.warning("Created minimal response with required fields")
                            return minimal_response
                    
                    return {"error": "Failed to parse response", "raw_text": response_text[:1000]}
            else:
                st.error("No JSON object found in response")
                st.code(f"Raw response:\n{response_text[:1000]}...")
                return {"error": "No JSON found", "raw_text": response_text[:1000]}
        else:
            # Standard text response with basic config
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# ------------------ MOLECULAR REPRESENTATION FUNCTIONS ------------------

def generate_molecule_from_description(
    description: str,
    show_chiral: bool = True
) -> Tuple[Optional[str], Optional[Tuple[Image.Image, Optional[Image.Image]]], Optional[Chem.Mol], bool]:
    """Generate a molecule from a natural language description with enhanced chiral visualization.
    
    Args:
        description: Natural language description of the molecule
        show_chiral: Whether to show mirror images for chiral molecules
        
    Returns:
        Tuple containing:
        - SMILES string or None if generation fails
        - Tuple of (standard image, Cairo image) or None if generation fails
        - RDKit molecule object or None if generation fails
        - Boolean indicating if the molecule is chiral
    """
    with st.spinner("Generating molecular structure..."):
        # Define JSON schema for molecular structure
        molecule_schema = {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Valid SMILES string representing the molecule with explicit stereochemistry"
                },
                "name": {
                    "type": "string",
                    "description": "Chemical name of the molecule"
                },
                "is_chiral": {
                    "type": "boolean",
                    "description": "Whether the molecule is chiral"
                },
                "description": {
                    "type": "string", 
                    "description": "Brief description of the molecule's structure and stereochemistry"
                }
            },
            "required": ["smiles", "name", "description"]
        }
        
        prompt = (
            f"Generate the molecular structure for: {description}\n\n"
            f"Include a valid SMILES string with explicit stereochemistry if the molecule is chiral.\n"
            f"For chiral molecules, use '@' or '@@' in SMILES to specify absolute stereochemistry.\n"
            f"Example: For bromochlorofluoromethane, use 'F[C@H](Cl)Br' or 'F[C@@H](Cl)Br'\n"
            f"IMPORTANT: Use only one '@' or '@@' symbol, never more.\n"
            f"Ensure all stereogenic centers are properly specified."
        )
        
        try:
            result = get_gemini_response(prompt, molecule_schema)
            
            if result and isinstance(result, dict) and "smiles" in result:
                smiles = result["smiles"].strip()
                # Clean up SMILES string - ensure proper @ usage
                smiles = re.sub(r'@{3,}', '@@', smiles)  # Replace 3 or more @ with @@
                smiles = re.sub(r'@{1}@(?!@)', '@', smiles)  # Replace @@ with @ if not @@@
                mol = Chem.MolFromSmiles(smiles)
                
                if mol:
                    # Check if molecule is chiral
                    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                    is_chiral = len(chiral_centers) > 0
                    
                    if is_chiral:
                        # Generate 2D coordinates with enhanced stereochemistry display
                        AllChem.Compute2DCoords(mol)
                        
                        # Create standard image first (480p)
                        img = Draw.MolToImage(mol, size=(640, 480))
                        
                        # Create Cairo image if available (480p)
                        cairo_img = None
                        if CAIRO_AVAILABLE:
                            drawer = MolDraw2DCairo(640, 480)  # 480p resolution
                            drawer.drawOptions().addStereoAnnotation = True
                            drawer.drawOptions().addAtomIndices = False
                            drawer.drawOptions().explicitMethyl = True
                            drawer.drawOptions().fixedBondLength = 40
                            drawer.drawOptions().fixedScale = 40
                            drawer.drawOptions().includeRadicals = True
                            drawer.drawOptions().additionalAtomLabelPadding = 0.2
                            drawer.drawOptions().bondLineWidth = 2.0
                            drawer.drawOptions().multipleBondOffset = 0.2
                            drawer.drawOptions().padding = 0.2
                            drawer.drawOptions().legendFontSize = 16
                            
                            drawer.DrawMolecule(mol)
                            drawer.FinishDrawing()
                            cairo_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                        
                        # If showing chiral pairs, generate mirror images
                        if show_chiral:
                            # Create mirror image by inverting stereochemistry
                            mirror_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                            mirror_smiles = mirror_smiles.replace('@', 'X').replace('@@', '@').replace('X', '@@')
                            mirror_mol = Chem.MolFromSmiles(mirror_smiles)
                            
                            if mirror_mol:
                                AllChem.Compute2DCoords(mirror_mol)
                                
                                # Create standard mirror image (480p)
                                mirror_img = Draw.MolToImage(mirror_mol, size=(640, 480))
                                
                                # Create Cairo mirror image if available (480p)
                                cairo_mirror_img = None
                                if CAIRO_AVAILABLE:
                                    mirror_drawer = MolDraw2DCairo(640, 480)
                                    mirror_drawer.drawOptions().addStereoAnnotation = True
                                    mirror_drawer.drawOptions().addAtomIndices = False
                                    mirror_drawer.drawOptions().explicitMethyl = True
                                    mirror_drawer.drawOptions().fixedBondLength = 40
                                    mirror_drawer.drawOptions().fixedScale = 40
                                    mirror_drawer.drawOptions().includeRadicals = True
                                    mirror_drawer.drawOptions().additionalAtomLabelPadding = 0.2
                                    mirror_drawer.drawOptions().bondLineWidth = 2.0
                                    mirror_drawer.drawOptions().multipleBondOffset = 0.2
                                    mirror_drawer.drawOptions().padding = 0.2
                                    mirror_drawer.drawOptions().legendFontSize = 16
                                    
                                    mirror_drawer.DrawMolecule(mirror_mol)
                                    mirror_drawer.FinishDrawing()
                                    cairo_mirror_img = Image.open(io.BytesIO(mirror_drawer.GetDrawingText()))
                                
                                # Create combined images for both standard and Cairo
                                # Standard combined image
                                combined_width = 1300  # Total width with margin
                                combined_height = 500  # Height with margin
                                combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
                                
                                # Calculate positions for centered placement
                                left_x = 25  # Left margin
                                right_x = 650  # Position for right image (with margin)
                                y_offset = 25  # Top margin
                                
                                # Paste images with margins
                                combined_img.paste(img, (left_x, y_offset))
                                combined_img.paste(mirror_img, (right_x, y_offset))
                                
                                # Draw mirror line with proper positioning
                                draw = ImageDraw.Draw(combined_img)
                                center_x = combined_width // 2
                                draw.line([(center_x, 10), (center_x, combined_height-10)], 
                                        fill='lightblue', width=2)
                                
                                # Add "Mirror" text with better font
                                try:
                                    font = ImageFont.truetype("arial.ttf", 16)
                                except:
                                    font = ImageFont.load_default()
                                
                                # Add mirror text with background
                                mirror_text = "Mirror Plane"
                                text_bbox = draw.textbbox((0, 0), mirror_text, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_x = center_x - text_width // 2
                                text_y = 10
                                
                                # Draw white background for text
                                padding = 5
                                draw.rectangle([
                                    text_x - padding,
                                    text_y - padding,
                                    text_x + text_width + padding,
                                    text_y + 20
                                ], fill='white')
                                
                                # Draw text
                                draw.text((text_x, text_y), mirror_text, 
                                        fill='black', font=font)
                                
                                # Create Cairo combined image if available
                                cairo_combined_img = None
                                if CAIRO_AVAILABLE and cairo_img and cairo_mirror_img:
                                    combined_width = 1300  # Total width with margin
                                    combined_height = 500  # Height with margin
                                    cairo_combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
                                    
                                    # Calculate positions for centered placement
                                    left_x = 25  # Left margin
                                    right_x = 650  # Position for right image (with margin)
                                    y_offset = 25  # Top margin
                                    
                                    # Paste images with margins
                                    cairo_combined_img.paste(cairo_img, (left_x, y_offset))
                                    cairo_combined_img.paste(cairo_mirror_img, (right_x, y_offset))
                                    
                                    # Draw mirror line
                                    draw = ImageDraw.Draw(cairo_combined_img)
                                    center_x = combined_width // 2
                                    draw.line([(center_x, 10), (center_x, combined_height-10)], 
                                            fill='lightblue', width=4)
                                    
                                    # Add mirror text with background
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 24)
                                    except:
                                        font = ImageFont.load_default()
                                    
                                    text_bbox = draw.textbbox((0, 0), mirror_text, font=font)
                                    text_width = text_bbox[2] - text_bbox[0]
                                    text_x = center_x - text_width // 2
                                    text_y = 20
                                    
                                    # Draw white background for text
                                    padding = 10
                                    draw.rectangle([
                                        text_x - padding,
                                        text_y - padding,
                                        text_x + text_width + padding,
                                        text_y + 30
                                    ], fill='white')
                                    
                                    # Draw text
                                    draw.text((text_x, text_y), mirror_text, 
                                            fill='black', font=font)
                                
                                return smiles, (combined_img, cairo_combined_img), mol, is_chiral
                            
                        return smiles, (img, cairo_img), mol, is_chiral
                    else:
                        # For non-chiral molecules, use standard visualization first (480p)
                        img = Draw.MolToImage(mol, size=(640, 480))
                        
                        # Create Cairo image if available (480p)
                        cairo_img = None
                        if CAIRO_AVAILABLE:
                            drawer = MolDraw2DCairo(640, 480)  # 480p resolution
                            drawer.drawOptions().addAtomIndices = False
                            drawer.drawOptions().explicitMethyl = True
                            drawer.drawOptions().fixedBondLength = 40
                            drawer.drawOptions().fixedScale = 40
                            drawer.drawOptions().bondLineWidth = 2.0
                            drawer.drawOptions().multipleBondOffset = 0.2
                            drawer.drawOptions().padding = 0.2
                            drawer.drawOptions().legendFontSize = 16
                            
                            drawer.DrawMolecule(mol)
                            drawer.FinishDrawing()
                            cairo_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                            
                            # Add white margin to Cairo image
                            margin = 50
                            cairo_img_with_margin = Image.new('RGB', 
                                                      (cairo_img.width + 2*margin, cairo_img.height + 2*margin), 
                                                      'white')
                            cairo_img_with_margin.paste(cairo_img, (margin, margin))
                            cairo_img = cairo_img_with_margin
                        
                        # Add white margin to standard image
                        margin = 50
                        img_with_margin = Image.new('RGB', 
                                                  (img.width + 2*margin, img.height + 2*margin), 
                                                  'white')
                        img_with_margin.paste(img, (margin, margin))
                        
                        return smiles, (img_with_margin, cairo_img), mol, False
                
                st.warning(f"Invalid SMILES string returned: {smiles}")
            else:
                st.error("Failed to generate structured molecule data")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    return None, None, None, False

def generate_3d_structure(mol: Chem.Mol) -> Optional[py3Dmol.view]:
    """Generate a 3D structure for the molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Py3DMol viewer object or None if generation fails
    """
    if not mol or not _3D_VISUALIZATION_AVAILABLE:
        return None
    
    try:
        # Create a 3D molecule
        mol_3d = Chem.AddHs(mol)
        
        # Check for metals that might cause UFF/MMFF issues
        has_metals = check_if_metal_atoms(mol_3d)
        
        # Generate 3D coordinates
        if has_metals:
            # For metal-containing molecules, use simpler embedding without optimization
            AllChem.EmbedMolecule(mol_3d, randomSeed=42, useRandomCoords=True)
        else:
            # For regular molecules, use the standard approach with optimization
            result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            
            if result == -1:
                # If standard embedding fails, try with more iterations
                params = AllChem.ETKDG()
                params.randomSeed = 42
                params.maxIterations = 2000
                params.numThreads = 4
                result = AllChem.EmbedMolecule(mol_3d, params)
            
            # Only optimize if embedding was successful and no metals are present
            if result == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_3d)
                except:
                    # If MMFF fails, try UFF
                    try:
                        AllChem.UFFOptimizeMolecule(mol_3d)
                    except:
                        # If both fail, just use the embedded coordinates without optimization
                        pass
        
        # Convert molecule to PDB string
        pdb_string = Chem.MolToPDBBlock(mol_3d)
        
        # Create 3D visualization
        viewer = py3Dmol.view(width=500, height=400)
        viewer.addModel(pdb_string, 'pdb')
        viewer.setStyle({'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.2}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        viewer.spin(True)
        
        return viewer
    except Exception as e:
        st.warning(f"Could not generate 3D structure: {str(e)}")
    
    return None

def display_molecular_properties(mol: Chem.Mol) -> None:
    """Display molecular properties in the UI.
    
    Args:
        mol: RDKit molecule object
    """
    if not mol:
        return
    
    # Calculate basic properties
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact_mass = Descriptors.ExactMolWt(mol)
    mol_weight = Descriptors.MolWt(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Display properties in a nice format
    st.subheader("Molecular Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Molecular Formula", formula)
        st.metric("Molecular Weight", f"{mol_weight:.2f}")
        st.metric("Exact Mass", f"{exact_mass:.4f}")
        st.metric("Number of Atoms", num_atoms)
    
    with col2:
        st.metric("Number of Bonds", num_bonds)
        st.metric("Number of Rings", num_rings)
        st.metric("LogP", f"{logp:.2f}")
        st.metric("TPSA", f"{tpsa:.2f}")

def generate_sd_file_download_link(mol: Chem.Mol, smiles: str, filename: str = "molecule.sdf") -> str:
    """Generate a download link for an SD file of the molecule.
    
    Args:
        mol: RDKit molecule object
        smiles: SMILES string of the molecule
        filename: Name for the downloaded file
        
    Returns:
        HTML link for downloading the SD file
    """
    # Create an SDWriter to a string
    sio = io.StringIO()
    writer = Chem.SDWriter(sio)
    
    # Add properties to the molecule
    mol.SetProp("SMILES", smiles)
    mol.SetProp("Formula", rdMolDescriptors.CalcMolFormula(mol))
    
    writer.write(mol)
    writer.close()
    
    # Get the string value and encode
    sdf_str = sio.getvalue()
    b64 = base64.b64encode(sdf_str.encode()).decode()
    
    return f'<a href="data:chemical/x-mdl-sdfile;base64,{b64}" download="{filename}">Download SD File</a>'

# ------------------ CHEMICAL EQUATION FLOW TAB ------------------

def generate_reaction_diagram(description, use_cairo=False):
    """Generate a reaction diagram from a description using structured JSON responses"""
    with st.spinner("Generating reaction diagram..."):
        # Define JSON schema for reaction responses
        reaction_schema = {
            "type": "object",
            "properties": {
                "balanced_equation": {
                    "type": "string",
                    "description": "The balanced chemical equation in text form"
                },
                "reaction_smarts": {
                    "type": "string", 
                    "description": "SMARTS notation for the reaction in format: reactant1.reactant2>>product1.product2"
                },
                "reactant_smiles": {
                    "type": "array",
                    "description": "Array of SMILES strings for each reactant",
                    "items": {"type": "string"}
                },
                "product_smiles": {
                    "type": "array", 
                    "description": "Array of SMILES strings for each product",
                    "items": {"type": "string"}
                },
                "mechanism": {
                    "type": "array",
                    "description": "Step-by-step explanation of the reaction mechanism",
                    "items": {"type": "string"}
                },
                "conditions": {
                    "type": "object",
                    "description": "Reaction conditions",
                    "properties": {
                        "temperature": {"type": "string"},
                        "pressure": {"type": "string"},
                        "catalyst": {"type": "string"},
                        "solvent": {"type": "string"},
                        "other": {"type": "string"}
                    }
                }
            },
            "required": ["balanced_equation", "reactant_smiles", "product_smiles", "mechanism", "conditions"]
        }
        
        prompt = (
            f"You are a chemistry expert. Create a detailed reaction mechanism diagram for the following chemical process:\n\n"
            f"{description}\n\n"
            f"Your response must include:\n"
            f"1. The balanced chemical equation in standard notation\n"
            f"2. SMILES strings for all reactants and products (provided separately, not as a reaction string)\n" 
            f"3. A step-by-step mechanism explanation (keep each step brief)\n"
            f"4. Reaction conditions (temperature, pressure, catalyst, solvent, etc.)\n\n"
            f"IMPORTANT FOR SMILES STRINGS:\n"
            f"- Provide individual, valid SMILES strings for each reactant and product\n"
            f"- Include all atoms and bonds explicitly in the SMILES\n"
            f"- Use proper aromatic notation (lowercase) where appropriate\n"
            f"- Include stereochemistry if relevant\n"
            f"- Separate multiple reactants/products into individual SMILES strings in arrays\n"
            f"- For the example reaction of salicylic acid with acetic anhydride:\n"
            f"  * Salicylic acid: c1ccccc1C(=O)O\n"
            f"  * Acetic anhydride: CC(=O)OC(=O)C\n"
            f"  * Aspirin: CC(=O)Oc1ccccc1C(=O)O\n"
            f"  * Acetic acid: CC(=O)O"
        )
        
        try:
            # Get structured response
            result = get_gemini_response(prompt, reaction_schema)
            
            if result and isinstance(result, dict):
                # Check if we have valid reactant and product SMILES
                reactant_smiles = result.get("reactant_smiles", [])
                product_smiles = result.get("product_smiles", [])
                
                # Generate combined reaction SMILES using the array approach
                reaction_smiles = ""
                valid_reactants = []
                valid_products = []
                
                # Try to validate each SMILES and generate 2D coordinates
                if isinstance(reactant_smiles, list) and len(reactant_smiles) > 0:
                    for smiles in reactant_smiles:
                        try:
                            if isinstance(smiles, str) and smiles.strip():
                                mol = Chem.MolFromSmiles(smiles.strip())
                                if mol:
                                    # Generate 2D coordinates
                                    AllChem.Compute2DCoords(mol)
                                    valid_reactants.append((mol, Chem.MolToSmiles(mol)))
                                else:
                                    st.warning(f"Invalid reactant SMILES: {smiles}")
                        except Exception as e:
                            st.warning(f"Error parsing reactant SMILES: {str(e)}")
                
                if isinstance(product_smiles, list) and len(product_smiles) > 0:
                    for smiles in product_smiles:
                        try:
                            if isinstance(smiles, str) and smiles.strip():
                                mol = Chem.MolFromSmiles(smiles.strip())
                                if mol:
                                    # Generate 2D coordinates
                                    AllChem.Compute2DCoords(mol)
                                    valid_products.append((mol, Chem.MolToSmiles(mol)))
                                else:
                                    st.warning(f"Invalid product SMILES: {smiles}")
                        except Exception as e:
                            st.warning(f"Error parsing product SMILES: {str(e)}")
                
                # Construct reaction SMILES from validated components
                if valid_reactants and valid_products:
                    reaction_smiles = ".".join([s for _, s in valid_reactants]) + ">>" + ".".join([s for _, s in valid_products])
                else:
                    # Fallback to reaction_smarts if available
                    reaction_smarts = result.get("reaction_smarts", "")
                    if reaction_smarts and ">>" in reaction_smarts:
                        reaction_smiles = reaction_smarts
                
                # Generate reaction visualization
                reaction_img = None
                cairo_img = None
                if reaction_smiles:
                    try:
                        # Try generating the reaction visualization
                        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
                        
                        # Validate the reaction has reactants and products
                        if rxn.GetNumReactantTemplates() > 0 and rxn.GetNumProductTemplates() > 0:
                            # Create standard image
                            reaction_img = Draw.ReactionToImage(rxn, subImgSize=(300, 200))
                            
                            # Create Cairo image if available
                            if use_cairo and CAIRO_AVAILABLE:
                                # Set up Cairo drawer with larger size
                                drawer = MolDraw2DCairo(900, 200)  # Wider to accommodate reaction arrow
                                drawer.drawOptions().prepareMolsBeforeDrawing = True
                                drawer.drawOptions().addAtomIndices = False
                                drawer.drawOptions().addBondIndices = False
                                
                                # Draw reaction with Cairo
                                drawer.DrawReaction(rxn)
                                drawer.FinishDrawing()
                                cairo_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                        else:
                            st.warning("Generated reaction doesn't have both reactants and products")
                    except Exception as e:
                        st.warning(f"Could not create reaction image: {str(e)}")
                
                # If we still don't have an image, try creating individual molecules
                if not reaction_img:
                    try:
                        all_mols = [mol for mol, _ in valid_reactants + valid_products]
                        if all_mols:
                            # Create standard grid image
                            reaction_img = Draw.MolsToGridImage(
                                all_mols,
                                molsPerRow=len(all_mols),
                                subImgSize=(300, 250),
                                legends=[f"R{i+1}" if i < len(valid_reactants) else f"P{i+1-len(valid_reactants)}" 
                                        for i in range(len(all_mols))]
                            )
                            
                            # Create Cairo grid image if available
                            if use_cairo and CAIRO_AVAILABLE:
                                img_size = (300 * len(all_mols), 250)
                                drawer = MolDraw2DCairo(*img_size)
                                drawer.DrawMolecules(
                                    all_mols,
                                    legends=[f"R{i+1}" if i < len(valid_reactants) else f"P{i+1-len(valid_reactants)}" 
                                            for i in range(len(all_mols))]
                                )
                                drawer.FinishDrawing()
                                cairo_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                    except Exception as e:
                        st.warning(f"Failed to create molecular grid: {str(e)}")
                
                # Format a nice textual explanation from the structured data
                formatted_explanation = f"# {description}\n\n"
                formatted_explanation += f"## Balanced Equation\n{result['balanced_equation']}\n\n"
                
                if reaction_smiles:
                    formatted_explanation += f"## Reaction SMILES\n`{reaction_smiles}`\n\n"
                
                formatted_explanation += "## Reactants\n"
                for i, (_, smiles) in enumerate(valid_reactants):
                    formatted_explanation += f"{i+1}. `{smiles}`\n"
                
                formatted_explanation += "\n## Products\n"
                for i, (_, smiles) in enumerate(valid_products):
                    formatted_explanation += f"{i+1}. `{smiles}`\n"
                
                formatted_explanation += "\n## Mechanism\n"
                if "mechanism" in result and isinstance(result["mechanism"], list):
                    for i, step in enumerate(result["mechanism"]):
                        formatted_explanation += f"{i+1}. {step}\n"
                
                formatted_explanation += "\n## Conditions\n"
                if "conditions" in result and isinstance(result["conditions"], dict):
                    for condition, value in result["conditions"].items():
                        if value and value != "null":
                            formatted_explanation += f"- **{condition.capitalize()}**: {value}\n"
                
                return formatted_explanation, reaction_smiles, reaction_img, cairo_img
            else:
                st.error("Failed to generate structured reaction data")
                if isinstance(result, dict) and "error" in result:
                    st.code(result.get("raw_text", "No error details available"))
                return None, None, None, None
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    return None, None, None, None

# ------------------ EQUATION-BASED QUESTION GENERATION TAB ------------------

def generate_chemistry_questions(context, num_questions, use_cairo=False):
    """Generate chemistry questions with a flexible number of molecular structure-based questions."""
    with st.spinner(f"Generating {num_questions} chemistry questions..."):
        # Define the JSON schema for chemistry questions
        questions_schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "Array of chemistry questions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full text of the chemistry question"
                            },
                            "equations": {
                                "type": "array",
                                "description": "List of equations relevant to the question. Use LaTeX formatting for chemical equations. Make sure to escape backslashes properly, e.g., \\frac{a}{b} for fractions, \\rightarrow for arrows, etc.",
                                "items": {"type": "string"}
                            },
                            "solution": {
                                "type": "string",
                                "description": "Detailed step-by-step solution to the question. Each step should be on a new line starting with 'Step X:'. Use LaTeX formatting for equations and special characters."
                            },
                            "smiles": {
                                "type": "string",
                                "description": "SMILES representation of a molecule (only if molecular structure is required to answer the question)"
                            },
                            "requires_molecular_structure": {
                                "type": "boolean",
                                "description": "Whether this question requires a molecular structure to be answered"
                            }
                        },
                        "required": ["question", "equations", "solution", "requires_molecular_structure"]
                    }
                }
            },
            "required": ["questions"]
        }
        
        prompt = (
            f"You are a chemistry teacher creating questions for students. "
            f"Based on the following context, generate {num_questions} chemistry questions.\n\n"
            f"Context: {context}\n\n"
            f"IMPORTANT GUIDELINES:\n"
            f"1. Create a mix of question types:\n"
            f"   - At least one question MUST involve molecular structure (set requires_molecular_structure: true)\n"
            f"   - For structure questions, ALWAYS provide a valid SMILES string\n"
            f"   - The remaining questions can be equation-based\n"
            f"2. For molecular structure questions:\n"
            f"   - REQUIRED: Include a valid SMILES string in the 'smiles' field\n"
            f"   - Example SMILES: 'CC(=O)Oc1ccccc1C(=O)O' for aspirin\n"
            f"   - Focus on structure-property relationships, functional groups, or reaction mechanisms\n"
            f"   - Questions should explicitly reference the molecular structure\n"
            f"3. For equation-based questions:\n"
            f"   - Focus on stoichiometry, equilibrium, kinetics, or thermodynamics\n"
            f"   - Include balanced chemical equations where relevant\n"
            f"   - Format equations using proper LaTeX with double backslashes (e.g., \\\\frac{{a}}{{b}} instead of \\frac{a}{b})\n"
            f"4. Keep all questions challenging but solvable\n"
            f"5. Format solutions with clear steps:\n"
            f"   - Each step should be on a new line\n"
            f"   - Start each step with 'Step X:' where X is the step number\n"
            f"   - Use LaTeX formatting for equations\n"
            f"6. Ensure each question is self-contained and clear\n"
            f"7. Mark each question with 'requires_molecular_structure': true/false\n"
            f"8. Keep all text fields under 500 characters\n"
            f"9. Do not use line breaks in string values\n"
            f"10. Use proper JSON formatting with double quotes\n"
            f"11. Use LaTeX formatting for all special characters and equations\n"
            f"12. IMPORTANT: Your response must be a complete, valid JSON object with all questions\n"
            f"13. Do not truncate or cut off the response\n"
            f"14. Ensure all arrays and objects are properly closed\n"
        )
        
        try:
            # Call Gemini with structured output schema
            result = get_gemini_response(prompt, questions_schema)
            
            if result and isinstance(result, dict):
                # Validate the response structure
                if "questions" not in result:
                    st.error("Response missing 'questions' array")
                    return None, None
                
                if not isinstance(result["questions"], list):
                    st.error("'questions' field is not an array")
                    return None, None
                
                # Validate each question has required fields
                valid_questions = []
                for i, q in enumerate(result["questions"]):
                    if not isinstance(q, dict):
                        st.warning(f"Question {i+1} is not a valid object")
                        continue
                    
                    # Check required fields
                    missing_fields = []
                    for field in ["question", "equations", "solution", "requires_molecular_structure"]:
                        if field not in q:
                            missing_fields.append(field)
                    
                    # Check for SMILES if molecular structure is required
                    if q.get("requires_molecular_structure", False) and (
                        "smiles" not in q or not q["smiles"] or q["smiles"] == "null"
                    ):
                        missing_fields.append("smiles")
                    
                    if missing_fields:
                        st.warning(f"Question {i+1} missing required fields: {', '.join(missing_fields)}")
                        continue
                    
                    # Validate equations array
                    if not isinstance(q["equations"], list):
                        q["equations"] = [q["equations"]]
                    
                    # Clean up SMILES string if present
                    if "smiles" in q:
                        q["smiles"] = q["smiles"].strip().replace('`', '')
                    
                    # Use Pydantic model to validate and serialize
                    try:
                        question_model = ChemistryQuestion(**q)
                        valid_questions.append(question_model)
                    except Exception as e:
                        st.warning(f"Validation error for question {i+1}: {str(e)}")
                        continue
                
                if not valid_questions:
                    st.error("No valid questions found in response")
                    return None, None
                
                # Ensure at least one but less than half of the questions are molecular structure-based
                num_structure_questions = sum(q.requires_molecular_structure for q in valid_questions)
                if num_structure_questions < 1 or num_structure_questions >= num_questions / 2:
                    st.warning("The distribution of question types is not balanced. Retrying...")
                    return generate_chemistry_questions(context, num_questions, use_cairo)
                
                # Serialize questions using Pydantic
                questions_json = [q.dict() for q in valid_questions]
                
                # Create a questions_data dictionary to return
                questions_data = {"questions": questions_json}
                
                # Now generate molecular images for questions that require them
                for question in questions_data["questions"]:
                    if question.get("requires_molecular_structure", False) and "smiles" in question:
                        # Convert SMILES to molecule
                        smiles = question["smiles"]
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Generate 2D coordinates
                            AllChem.Compute2DCoords(mol)
                            
                            # Create standard image
                            img = Draw.MolToImage(mol, size=(640, 480))
                            question["standard_image"] = image_to_base64(img)
                            
                            # Create Cairo image if available
                            if use_cairo and CAIRO_AVAILABLE:
                                drawer = MolDraw2DCairo(640, 480)
                                drawer.DrawMolecule(mol)
                                drawer.FinishDrawing()
                                cairo_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                                question["cairo_image"] = image_to_base64(cairo_img)
                                question["image_type"] = "both"
                
                return questions_data, "Structured response successful"
            else:
                st.error("Failed to generate structured questions data")
                if isinstance(result, dict) and "raw_text" in result:
                    st.code(result["raw_text"])
                return None, None
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    return None, None

# ------------------ UI COMPONENTS ------------------

def create_molecular_representation_tab() -> None:
    """Create and handle the Molecular Representation tab."""
    st.header("Molecular Representation")
    st.markdown("""
    Enter a description of a molecule, and the system will generate a corresponding molecular diagram.
    """)
    
    with st.form("molecule_form"):
        user_description = st.text_area(
            "Describe the molecule you want to visualize:",
            placeholder="Example: A benzene ring with a carboxylic acid group and a chlorine atom",
            height=100
        )
        
        examples = st.selectbox(
            "Or choose from examples:",
            [
                "None",
                "Aspirin (acetylsalicylic acid)",
                "Caffeine",
                "Paracetamol (acetaminophen)",
                "Ibuprofen",
                "Penicillin G",
                "Adrenaline (epinephrine)",
                "Glucose",
                "Cholesterol",
                "Vitamin C (ascorbic acid)",
                "ATP (adenosine triphosphate)"
            ]
        )
        
        view_options = st.columns(3)
        with view_options[0]:
            show_2d = st.checkbox("Show 2D Structure", value=True)
        with view_options[1]:
            show_3d = st.checkbox("Show 3D Structure", value=_3D_VISUALIZATION_AVAILABLE)
        with view_options[2]:
            use_cairo = st.checkbox("Use Cairo Rendering", value=CAIRO_AVAILABLE, disabled=not CAIRO_AVAILABLE)
        
        # Add checkbox for mirroring chiral structures
        show_mirror = st.checkbox("Show Mirror Image for Chiral Structures", value=False)
        
        submitted_mol = st.form_submit_button("Generate Molecule")
    
    # Handle example selection
    if examples != "None" and submitted_mol:
        user_description = examples
    
    # Process the submission
    if submitted_mol and (user_description and user_description != "None"):
        smiles, images, mol, is_chiral = generate_molecule_from_description(user_description, show_chiral=show_mirror)
        
        if smiles and images and mol:
            structure_tab, properties_tab, export_tab = st.tabs(["Molecular Structure", "Properties", "Export"])
            
            with structure_tab:
                if is_chiral:
                    st.subheader("Chiral Structure")
                    
                    # Display standard rendering
                    standard_img, cairo_img = images
                    st.markdown("#### Standard Rendering")
                    st.image(standard_img, use_column_width=True)
                    
                    # Display Cairo rendering if available
                    if cairo_img and use_cairo:
                        st.markdown("#### Enhanced Cairo Rendering")
                        st.image(cairo_img, use_column_width=True)
                    
                    if show_mirror:
                        st.info("This molecule is chiral. The images show the molecule and its mirror image (enantiomer).")
                    else:
                        st.info("This molecule is chiral. The images show the molecule in its current stereochemistry.")
                else:
                    st.subheader("2D Structure")
                    
                    # Display standard rendering
                    standard_img, cairo_img = images
                    st.markdown("#### Standard Rendering")
                    st.image(standard_img, width=400)
                    
                    # Display Cairo rendering if available
                    if cairo_img and use_cairo:
                        st.markdown("#### Enhanced Cairo Rendering")
                        st.image(cairo_img, width=400)
                
                if show_3d and _3D_VISUALIZATION_AVAILABLE:
                    st.subheader("3D Structure")
                    viewer_3d = generate_3d_structure(mol)
                    if viewer_3d:
                        showmol(viewer_3d, height=500, width=800)
                    else:
                        st.warning("3D structure visualization is not available for this molecule.")
            
            with properties_tab:
                display_molecular_properties(mol)
                
                # Display atom details
                st.subheader("Atom Details")
                atom_data = []
                for atom in mol.GetAtoms():
                    atom_data.append({
                        "Index": atom.GetIdx(),
                        "Element": atom.GetSymbol(),
                        "Formal Charge": atom.GetFormalCharge(),
                        "Hybridization": str(atom.GetHybridization()),
                        "Is Aromatic": atom.GetIsAromatic(),
                        "Is In Ring": atom.IsInRing()
                    })
                
                if atom_data:
                    st.dataframe(atom_data)
            
            with export_tab:
                # Add download options
                st.subheader("Download Options")
                
                # Image download
                img_bytes = io.BytesIO()
                standard_img.save(img_bytes, format='PNG')
                st.download_button(
                    label="Download Standard Diagram (PNG)",
                    data=img_bytes.getvalue(),
                    file_name="molecule_standard.png",
                    mime="image/png"
                )
                
                if cairo_img:
                    img_bytes = io.BytesIO()
                    cairo_img.save(img_bytes, format='PNG')
                    st.download_button(
                        label="Download Cairo Diagram (PNG)",
                        data=img_bytes.getvalue(),
                        file_name="molecule_cairo.png",
                        mime="image/png"
                    )
                
                # SDF download 
                st.markdown(generate_sd_file_download_link(mol, smiles), unsafe_allow_html=True)
                
                # SMILES text download
                st.download_button(
                    label="Download SMILES",
                    data=smiles,
                    file_name="molecule.smi",
                    mime="text/plain"
                )
        else:
            st.error("Failed to generate a valid molecular structure from the description. Please try a different description or be more specific.")

def create_chemical_equation_tab() -> None:
    """Create and handle the Chemical Equation Flow tab."""
    st.header("Chemical Equation Flow")
    st.markdown("""
    Enter a description of a chemical reaction, and the system will generate a complete reaction flow diagram with an explanation.
    """)
    
    with st.form("reaction_form"):
        reaction_description = st.text_area(
            "Describe the chemical reaction or process:",
            placeholder="Example: The hydrolysis of ethyl acetate in basic conditions",
            height=100
        )
        
        reaction_examples = st.selectbox(
            "Or choose from examples:",
            [
                "None",
                "Synthesis of aspirin from salicylic acid and acetic anhydride",
                "Esterification of acetic acid with ethanol",
                "Aldol condensation of acetone",
                "Diels-Alder reaction between cyclopentadiene and maleic anhydride",
                "SN2 reaction of bromide with hydroxide",
                "Grignard reaction with benzaldehyde",
                "Acetylation of phenol with acetic anhydride"
            ]
        )
        
        view_options = st.columns(2)
        with view_options[0]:
            show_mechanism = st.checkbox("Show Mechanism Steps", value=True)
        with view_options[1]:
            use_cairo = st.checkbox("Use Cairo Rendering", value=CAIRO_AVAILABLE, disabled=not CAIRO_AVAILABLE)
        
        submitted_reaction = st.form_submit_button("Generate Reaction Diagram")
    
    # Handle example selection
    if reaction_examples != "None" and submitted_reaction:
        reaction_description = reaction_examples
    
    # Process the submission
    if submitted_reaction and (reaction_description and reaction_description != "None"):
        explanation, reaction_smiles, reaction_img, cairo_img = generate_reaction_diagram(reaction_description, use_cairo)
        
        if explanation:
            # Display results in tabs
            mechanism_tab, visualization_tab, export_tab = st.tabs(["Reaction Mechanism", "Visualization", "Export"])
            
            with mechanism_tab:
                if show_mechanism:
                    st.markdown(explanation)
                else:
                    # Show only the balanced equation and conditions
                    lines = explanation.split('\n')
                    filtered_lines = []
                    show_line = True
                    for line in lines:
                        if '## Mechanism' in line:
                            show_line = False
                        elif '## Conditions' in line:
                            show_line = True
                        if show_line:
                            filtered_lines.append(line)
                    st.markdown('\n'.join(filtered_lines))
            
            with visualization_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    if reaction_img:
                        st.subheader("Standard Rendering")
                        st.image(reaction_img, use_column_width=True)
                
                with col2:
                    if cairo_img and use_cairo:
                        st.subheader("Cairo Rendering")
                        st.image(cairo_img, use_column_width=True)
                
                if reaction_smiles:
                    st.subheader("Reaction SMILES")
                    st.code(reaction_smiles)
            
            with export_tab:
                st.subheader("Download Options")
                
                # Reaction explanation download
                st.download_button(
                    label="Download Explanation (Text)",
                    data=explanation,
                    file_name="reaction_explanation.txt",
                    mime="text/plain"
                )
                
                # Standard image download
                if reaction_img:
                    img_bytes = io.BytesIO()
                    reaction_img.save(img_bytes, format='PNG')
                    st.download_button(
                        label="Download Standard Diagram (PNG)",
                        data=img_bytes.getvalue(),
                        file_name="reaction_diagram_standard.png",
                        mime="image/png"
                    )
                
                # Cairo image download
                if cairo_img and use_cairo:
                    img_bytes = io.BytesIO()
                    cairo_img.save(img_bytes, format='PNG')
                    st.download_button(
                        label="Download Cairo Diagram (PNG)",
                        data=img_bytes.getvalue(),
                        file_name="reaction_diagram_cairo.png",
                        mime="image/png"
                    )
        else:
            st.error("Failed to generate the reaction diagram. Please try a different description or be more specific.")

def create_question_generation_tab() -> None:
    """Create and handle the Equation-Based Question Generation tab."""
    st.header("Equation-Based Question Generation")
    st.markdown("""
    Provide a chemistry context and specify the number of questions to generate. The system will create equation-based chemistry questions with solutions.
    """)
    
    with st.form("question_form"):
        context = st.text_area(
            "Enter the chemistry context for questions:",
            placeholder="Example: Acid-base equilibria and pH calculations in buffer solutions",
            height=100
        )
        
        num_questions = st.number_input(
            "Number of questions to generate:",
            min_value=1,
            max_value=5,
            value=3
        )
        
        context_examples = st.selectbox(
            "Or choose from example contexts:",
            [
                "None",
                "Organic chemistry reactions and mechanisms",
                "Ideal gas law and gas phase reactions",
                "Redox reactions and electrochemistry",
                "Chemical kinetics and reaction rates",
                "Thermodynamics and enthalpy calculations",
                "Acid-base equilibria and buffer solutions",
                "Nuclear chemistry and radioactive decay"
            ]
        )
        
        view_options = st.columns(2)
        with view_options[0]:
            show_solutions = st.checkbox("Show Solutions", value=True)
        with view_options[1]:
            use_cairo = st.checkbox("Use Cairo Rendering", value=CAIRO_AVAILABLE, disabled=not CAIRO_AVAILABLE)
        
        submitted_questions = st.form_submit_button("Generate Questions")
    
    # Handle example selection
    if context_examples != "None" and submitted_questions:
        context = context_examples
    
    # Process the submission
    if submitted_questions and (context and context != "None"):
        questions_data, raw_response = generate_chemistry_questions(context, num_questions, use_cairo)
        
        if questions_data and "questions" in questions_data:
            st.success(f"Generated {len(questions_data['questions'])} chemistry questions!")
            
            # Create an expandable section for each question
            for i, question in enumerate(questions_data["questions"]):
                with st.expander(f"Question {i+1}: {question['question'][:100]}...", expanded=(i == 0)):
                    st.markdown("### Question")
                    st.markdown(question["question"])
                    
                    st.markdown("### Equations")
                    for eq in question.get("equations", []):
                        # Fix LaTeX slashes before rendering
                        fixed_eq = fix_latex_slashes(eq)
                        try:
                            st.latex(fixed_eq)
                        except Exception as e:
                            st.error(f"Error rendering equation: {str(e)}")
                            st.code(f"Raw equation: {eq}")
                            st.code(f"Fixed equation: {fixed_eq}")
                    
                    if question.get("requires_molecular_structure", False):
                        st.markdown("### Molecular Structure")
                        
                        # Create a row with text and images side by side
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("The molecular structure for this question is shown below:")
                            if "smiles" in question:
                                st.markdown("**SMILES String:**")
                                st.code(question["smiles"])
                        
                        with col2:
                            # Display standard image
                            if "standard_image" in question:
                                st.image(f"data:image/png;base64,{question['standard_image']}", width=200)
                            
                            # Display Cairo image if available
                            if "cairo_image" in question and use_cairo and question.get("image_type") == "both":
                                st.image(f"data:image/png;base64,{question['cairo_image']}", width=200)
                            elif use_cairo and not CAIRO_AVAILABLE:
                                st.info("Cairo rendering not available")
                    
                    if show_solutions:
                        st.markdown("### Solution")
                        # Split solution into steps and render each step with LaTeX
                        steps = question["solution"].split("\n")
                        for step in steps:
                            if step.strip():  # Only process non-empty lines
                                if step.startswith("Step"):
                                    # Display step header
                                    st.markdown(f"**{step}**")
                                else:
                                    # Check if the line contains LaTeX symbols
                                    if any(symbol in step for symbol in ['\\', '_', '^', '=']) or any(pattern in step for pattern in ['frac', 'rightarrow', 'leftrightarrow', 'alpha', 'beta', 'delta']):
                                        # Fix LaTeX slashes
                                        step = fix_latex_slashes(step)
                                        # Render LaTeX expression
                                        try:
                                            st.latex(step.strip())
                                        except Exception as e:
                                            st.error(f"Error rendering solution step: {str(e)}")
                                            st.markdown(step)  # Fallback to regular markdown
                                    else:
                                        # Regular text
                                        st.markdown(step)
            
            # Download options
            st.subheader("Download Options")
            
            # Download as JSON
            json_str = json.dumps(questions_data, indent=2)
            st.download_button(
                label="Download Questions (JSON)",
                data=json_str,
                file_name="chemistry_questions.json",
                mime="application/json"
            )
            
            # Download as formatted text
            formatted_text = ""
            for i, q in enumerate(questions_data["questions"]):
                formatted_text += f"Question {i+1}:\n{q['question']}\n\n"
                formatted_text += "Equations:\n"
                for eq in q.get("equations", []):
                    formatted_text += f"{eq}\n"
                if show_solutions:
                    formatted_text += f"\nSolution:\n{q['solution']}\n"
                formatted_text += "-" * 50 + "\n\n"
            
            st.download_button(
                label="Download Questions (Text)",
                data=formatted_text,
                file_name="chemistry_questions.txt",
                mime="text/plain"
            )
        else:
            st.error("Failed to generate questions. Please try a different context or be more specific.")

# ------------------ MAIN APPLICATION ------------------

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Molecular Representation", 
    "Chemical Equation Flow", 
    "Equation-Based Questions"
])

# Handle each tab
with tab1:
    create_molecular_representation_tab()

with tab2:
    create_chemical_equation_tab()

with tab3:
    create_question_generation_tab()

# Footer
st.markdown("""
---
This application uses:
- [RDKit](https://www.rdkit.org/) for molecular operations and visualization
- [Google's Gemini AI](https://ai.google.dev/) for generating chemical content
- [Streamlit](https://streamlit.io/) for the web interface
- [Py3DMol](https://3dmol.csb.pitt.edu/) for 3D molecular visualization (when available)
""")

