import ast
import os
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from .call_llm import semantic_analizes

@dataclass
class RelationshipScore:
    direct_import: float = 0.0  # Direct import relationships
    inheritance: float = 0.0    # Class inheritance relationships  
    function_calls: float = 0.0 # Function/method calls between files
    semantic: float = 0.0       # Semantic similarity score
    
    def total(self) -> float:
        """Calculate weighted total score."""
        weights = {
            'direct_import': 0.3,
            'inheritance': 0.3,
            'function_calls': 0.2,
            'semantic': 0.2
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

class CodeAnalyzer:
    def __init__(self):
        self._relationship_cache: Dict[str, Set[str]] = {}
        self._inheritance_cache: Dict[str, Set[str]] = {}
        self._score_cache: Dict[str, Dict[str, RelationshipScore]] = {}

    @lru_cache(maxsize=1000)
    def analyze_file_relationships(self, files_data: List[Tuple[str, str]],
                                 file_patterns: Set[str], 
                                 target_list: Set[str] = None) -> Dict[str, float]:
        """Analyze relationships between files and target list.
        
        Args:
            files_data: List of (path, content) tuples
            target_list: Set of target patterns to prioritize
            file_patterns: Set of file patterns to consider for relationships
            
        Returns:
            Dict mapping file paths to priority scores (0.0-1.0):
            - Direct target matches: 1.0
            - Direct relationships: 0.5-0.9 
            - Secondary relationships: 0.1-0.4
            - No relationship: 0.0
        """
        priorities = {}
        
        # First pass - static analysis
        for path, content in files_data:
            if any(target.lower() in path.lower() for target in target_list):
                # Direct target match
                priorities[path] = 1.0
                continue
                
            score = RelationshipScore()
            
            try:
                # Language-specific analysis
                if path.endswith('.py'):
                    # Python - Use AST
                    tree = ast.parse(content)
                    imports = self._extract_imports(tree)
                    score.direct_import = self._calculate_import_score(imports, target_list)
                    score.inheritance = self._analyze_inheritance(tree, files_data, target_list)
                    score.function_calls = self._analyze_function_calls(tree, files_data, target_list)
                elif path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    # JavaScript/TypeScript
                    imports = self._extract_js_imports(content)
                    score.direct_import = self._calculate_import_score(imports, target_list)
                elif path.endswith('.go'):
                    # Go
                    imports = self._extract_go_imports(content)
                    score.direct_import = self._calculate_import_score(imports, target_list)
                elif path.endswith('.java'):
                    # Java
                    imports = self._extract_java_imports(content)
                    score.direct_import = self._calculate_import_score(imports, target_list)
                    
                if not target_list:
                    score.semantic = 0.0
                else:
                    # Add semantic analysis for all file types
                    score.semantic = self._analyze_semantic_relationship(content, files_data, target_list)
                
                # Cache the score
                if path not in self._score_cache:
                    self._score_cache[path] = {}
                    
                for target in target_list:
                    self._score_cache[path][target] = score
                    
            except Exception as e:
                print(f"Warning: Error analyzing {path}: {e}")
                continue

        # Calculate final priorities            
        for path, scores in self._score_cache.items():
            if path not in priorities:  # Skip direct matches
                priorities[path] = max(score.total() for score in scores.values())
                
        return priorities
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from a Python AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        return imports
    
    def _extract_js_imports(self, content: str) -> Set[str]:
        """Extract imports from JavaScript/TypeScript code."""
        imports = set()
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('export '):
                if ' from ' in line:
                    module = line.split(' from ')[1].strip("'\"`;")
                    imports.add(module)
            elif 'require(' in line:
                start = line.find('require(') + 8
                end = line.find(')', start)
                if start > 7 and end > start:
                    module = line[start:end].strip("'\"")
                    imports.add(module)
        return imports
        
    def _extract_go_imports(self, content: str) -> Set[str]:
        """Extract imports from Go code."""
        imports = set()
        in_import_block = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ('):
                in_import_block = True
            elif line == ')':
                in_import_block = False
            elif line.startswith('import ') or in_import_block:
                parts = line.split()
                if len(parts) > 1:
                    package = parts[-1].strip('"')
                    imports.add(package)
        return imports
        
    def _extract_java_imports(self, content: str) -> Set[str]:
        """Extract imports from Java code."""
        imports = set()
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                parts = line.replace('import', '').replace('static', '').strip().rstrip(';').split('.')
                if parts:
                    imports.add(parts[0])
        return imports

    def _calculate_import_score(self, imports: Set[str], target_list: Set[str]) -> float:
        """Calculate import relationship score."""
        direct_matches = sum(any(target.lower() in imp.lower() 
                           for target in target_list)
                           for imp in imports)
        return min(0.8, direct_matches * 0.2)  # Cap at 0.8
        
    def _analyze_inheritance(self, tree: ast.AST, files_data: List[Tuple[str, str]], 
                           target_list: Set[str]) -> float:
        """Analyze class inheritance relationships."""
        score = 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        # Direct base class
                        if self._is_target_class(base.id, files_data, target_list):
                            score += 0.3
                    elif isinstance(base, ast.Attribute):
                        # Imported base class
                        if self._is_target_class(base.attr, files_data, target_list):
                            score += 0.2
        return min(0.9, score)  # Cap at 0.9
    
    def _analyze_function_calls(self, tree: ast.AST, files_data: List[Tuple[str, str]], 
                              target_list: Set[str]) -> float:
        """Analyze function call relationships."""
        score = 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if self._is_target_function(node.func.id, files_data, target_list):
                        score += 0.1
                elif isinstance(node, ast.Attribute):
                    if self._is_target_function(node.func.attr, files_data, target_list):
                        score += 0.1
        return min(0.7, score)  # Cap at 0.7

    def _analyze_semantic_relationship(self, content: str, files_data: List[Tuple[str, str]], 
                                     target_list: Set[str]) -> float:
        """Analyze semantic relationships using LLM.
        
        Uses an LLM to identify semantic relationships between files based on:
        - Shared domain concepts
        - Similar functionality
        - Architectural relationships
        - Business logic connections
        """
        if not target_list:
            return 0.0

        # Create context of target files
        target_context = ""
        for path, content in files_data:
            if any(target.lower() in path.lower() for target in target_list):
                target_context += f"\nFile: {path}\n{content}\n"

        if not target_context:
            return 0.0

        prompt = f"""Analyze the semantic relationship between the following code and the target files.
Focus on:
1. Shared domain concepts and terminology
2. Similar functionality or purpose
3. Architectural relationships
4. Business logic connections

Target files context:
{target_context}

Code to analyze:
{content}

Rate the semantic relationship strength from 0.0 to 0.4, where:
0.0 = No semantic relationship
0.1-0.2 = Weak semantic relationship (few shared concepts)
0.3-0.4 = Strong semantic relationship (many shared concepts)

Output only the number (e.g. "0.2")."""

        try:
            response = semantic_analizes(prompt)
            score = float(response.strip())
            return min(max(score, 0.0), 0.4)  # Clamp between 0.0 and 0.4
        except (ValueError, TypeError):
            return 0.0  # Default to no relationship on error

    def _is_target_class(self, class_name: str, files_data: List[Tuple[str, str]], 
                        target_list: Set[str]) -> bool:
        """Check if a class name belongs to a target module."""
        return any(target.lower() in class_name.lower() for target in target_list)
    
    def _is_target_function(self, func_name: str, files_data: List[Tuple[str, str]], 
                           target_list: Set[str]) -> bool:
        """Check if a function belongs to a target module."""
        return any(target.lower() in func_name.lower() for target in target_list)
    
    def _find_related_file(self, import_name: str, current_dir: str, all_files: List[str], file_patterns: Set[str] = None) -> Optional[str]:
        """Find the source file for an imported name across all supported file types.
        
        Args:
            import_name: The name to search for (e.g., module name)
            current_dir: Current directory context
            all_files: List of all files in the workspace
            file_patterns: Set of file patterns to consider (e.g., "*.py", "*.js")
        """
        if file_patterns is None:
            # Default to common source file extensions
            file_patterns = {
                "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java",
                "*.c", "*.cc", "*.cpp", "*.h"
            }
            
        # Extract extensions from patterns
        extensions = set()
        for pattern in file_patterns:
            if pattern.startswith("*."):
                extensions.add(pattern[1:])  # Remove the '*'
                
        # Generate potential paths for each extension
        potential_paths = []
        for ext in extensions:
            # Direct match with extension
            potential_paths.append(f"{import_name}{ext}")
            # Index file in directory
            potential_paths.extend([
                os.path.join(import_name, f"index{ext}"),
                os.path.join(import_name, f"main{ext}"),
                os.path.join(import_name, f"{os.path.basename(import_name)}{ext}")
            ])
            # Full path with extension
            potential_paths.append(os.path.join(current_dir, f"{import_name}{ext}"))
            
        # Special case for Python packages
        if "*.py" in file_patterns:
            potential_paths.extend([
                os.path.join(import_name, "__init__.py"),
                os.path.join(current_dir, import_name, "__init__.py")
            ])
            
        for file in all_files:
            if any(file.endswith(path) for path in potential_paths):
                return file
                
        return None
