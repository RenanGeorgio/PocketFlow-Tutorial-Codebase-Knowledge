from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TechnicalNote:
    content: str  # The detailed technical explanation
    references: List[str]  # References to specific code files/lines
    category: str  # Category of the note (e.g., "implementation", "api", "performance")
    tags: List[str]  # Additional metadata tags
    
class TechnicalNotesManager:
    def __init__(self):
        self.notes: Dict[str, List[TechnicalNote]] = {}  # Map abstraction name to its technical notes
        
    def add_note(self, abstraction_name: str, note: TechnicalNote):
        if abstraction_name not in self.notes:
            self.notes[abstraction_name] = []
        self.notes[abstraction_name].append(note)
        
    def get_notes(self, abstraction_name: str, category: Optional[str] = None) -> List[TechnicalNote]:
        """Get all technical notes for an abstraction, optionally filtered by category"""
        if abstraction_name not in self.notes:
            return []
        if category:
            return [note for note in self.notes[abstraction_name] if note.category == category]
        return self.notes[abstraction_name]
        
    def format_notes_as_markdown(self, abstraction_name: str) -> str:
        """Convert technical notes to Markdown format with proper sections"""
        if abstraction_name not in self.notes:
            return ""
            
        # Group notes by category
        notes_by_category = {}
        for note in self.notes[abstraction_name]:
            if note.category not in notes_by_category:
                notes_by_category[note.category] = []
            notes_by_category[note.category].append(note)
            
        # Build the markdown
        sections = []
        for category, notes in notes_by_category.items():
            section = [f"\n## Technical Deep Dive: {category.title()}\n"]
            for note in notes:
                section.append(note.content)
                if note.references:
                    section.append("\nRelevant code:")
                    section.append("```")
                    for ref in note.references:
                        section.append(ref)
                    section.append("```")
                if note.tags:
                    section.append(f"\nTags: {', '.join(note.tags)}")
                section.append("\n---\n")
            sections.append("\n".join(section))
            
        return "\n".join(sections)
