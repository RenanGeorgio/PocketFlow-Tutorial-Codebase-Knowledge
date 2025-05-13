from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import yaml
import re
from .call_llm import call_llm

@dataclass
class AnswerInfo:
    question: str                    # The original question
    full_answer: str                # Detailed answer content
    related_abstractions: List[str]  # Names of related abstractions
    snippets: Dict[str, str]        # Map of abstraction name to relevant snippet
    visual_markers: Dict[str, List[str]]  # Map of abstraction to visual markers/highlights
    references: Set[str]            # Set of related file paths
    tags: List[str]                 # Keywords/categories for the answer

class KnowledgeBaseManager:
    def __init__(self):
        self.answers: Dict[str, AnswerInfo] = {}  # Map question to answer info
        self.answer_pages: Dict[str, str] = {}    # Map answer ID to full answer page content
        
    def process_questions(self, questions: List[str], context: dict) -> None:
        """Process questions against the knowledge base context."""
        for question in questions:
            print(f"Processing question: {question}")
            
            # Generate detailed answer using accumulated context
            answer_prompt = self._create_answer_prompt(question, context)
            answer_response = call_llm(answer_prompt)
            
            # Analyze answer to determine related abstractions
            analysis_prompt = self._create_analysis_prompt(question, answer_response, context)
            analysis_response = call_llm(analysis_prompt)
            
            # Create answer info object with proper structure
            answer_info = self._parse_answer_analysis(question, answer_response, analysis_response)
            self.answers[question] = answer_info
            
            # Generate full answer page
            page_id = self._generate_page_id(question)
            page_content = self._create_answer_page(answer_info)
            self.answer_pages[page_id] = page_content
            
            print(f"Generated answer with {len(answer_info.related_abstractions)} related abstractions")

    def get_snippets_for_abstraction(self, abstraction_name: str) -> List[tuple[str, str, List[str]]]:
        """Get all answer snippets related to an abstraction.
        Returns list of (question, snippet, markers)."""
        snippets = []
        for question, answer_info in self.answers.items():
            if abstraction_name in answer_info.related_abstractions:
                snippet = answer_info.snippets.get(abstraction_name, "")
                markers = answer_info.visual_markers.get(abstraction_name, [])
                if snippet:
                    snippets.append((question, snippet, markers))
        return snippets

    def get_answer_pages(self) -> Dict[str, str]:
        """Get all generated answer pages."""
        return self.answer_pages

    def _create_answer_prompt(self, question: str, context: dict) -> str:
        """Create prompt to generate detailed answer."""
        context_str = self._format_context(context)
        return f"""Based on the following codebase context, provide an extremely detailed answer to this question:

Question: {question}

Relevant Context:
{context_str}

Generate a detailed, technical, and clear answer that:
1. Thoroughly explains all relevant aspects
2. References specific code and implementation details
3. Includes practical examples
4. Explains any important relationships or interactions
5. Highlights key technical considerations
6. Uses proper Markdown formatting with appropriate sections and code blocks
7. Makes extensive use of visual aids (diagrams, flowcharts) where helpful

Use the following markers for emphasis:
- 💡 For key insights and important points
- ⚡ For implementation details and technical notes
- 🔍 For deep dive explanations
- 📌 For related concepts and cross-references
- 🔗 For links to other parts of the codebase

Format the response in Markdown, and ensure all code examples are properly formatted in code blocks."""

    def _create_analysis_prompt(self, question: str, answer: str, context: dict) -> str:
        """Create prompt to analyze answer and determine related abstractions."""
        context_str = self._format_context(context)
        return f"""Analyze this question and answer in the context of the codebase:

Question: {question}

Answer: {answer}

Codebase Context:
{context_str}

For each relevant abstraction from the codebase, provide:
1. The name of the abstraction
2. A concise but informative snippet that should be shown in that abstraction's chapter
3. Visual markers/highlights to emphasize key points
4. References to specific files or code sections that are relevant

Format as YAML with this structure:
related_abstractions:
  - name: AbstractionName
    snippet: |
      A concise Markdown snippet that highlights how this abstraction relates to the answer.
      Include one or two key insights and link to the full answer.
    markers:
      - "💡 Key Point: Important insight about this abstraction"
      - "⚡ Implementation Note: Technical detail relevant to this abstraction"
      - "🔍 Deep Dive: Complex aspect that deserves attention"
      - "📌 Related Concept: Connection to other abstractions"
    references:
      - "path/to/relevant/file.py"
tags:
  - "relevant"
  - "technical"
  - "keywords"
"""

    def _parse_answer_analysis(self, question: str, answer: str, analysis_str: str) -> AnswerInfo:
        """Parse the YAML analysis response into an AnswerInfo object."""
        try:
            # Parse YAML response
            analysis = yaml.safe_load(analysis_str)
            
            # Extract information
            abstractions = []
            snippets = {}
            visual_markers = {}
            all_references = set()
            
            # Process each abstraction
            for abstraction in analysis.get("related_abstractions", []):
                name = abstraction["name"]
                abstractions.append(name)
                snippets[name] = abstraction.get("snippet", "")
                visual_markers[name] = abstraction.get("markers", [])
                all_references.update(abstraction.get("references", []))
            
            # Create AnswerInfo object
            return AnswerInfo(
                question=question,
                full_answer=answer,
                related_abstractions=abstractions,
                snippets=snippets,
                visual_markers=visual_markers,
                references=all_references,
                tags=analysis.get("tags", [])
            )
            
        except Exception as e:
            print(f"Error parsing answer analysis: {e}")
            # Return minimal AnswerInfo if parsing fails
            return AnswerInfo(
                question=question,
                full_answer=answer,
                related_abstractions=[],
                snippets={},
                visual_markers={},
                references=set(),
                tags=[]
            )

    def _generate_page_id(self, question: str) -> str:
        """Generate a valid filename from a question."""
        # Remove special characters and convert spaces to underscores
        safe_id = re.sub(r'[^\w\s-]', '', question.lower())
        safe_id = re.sub(r'[-\s]+', '_', safe_id)
        return f"answer_{safe_id[:50]}"  # Limit length and prefix with 'answer_'

    def _create_answer_page(self, answer_info: AnswerInfo) -> str:
        """Create a full answer page in Markdown format."""
        # Create page header
        content = [
            "---",
            "layout: default",
            "title: \"Q&A: " + answer_info.question.replace('"', '\\"') + "\"",
            "parent: \"Knowledge Base Q&A\"",
            "---",
            "",
            f"# {answer_info.question}",
            "",
            "## Detailed Answer",
            "",
            answer_info.full_answer,
            "",
            "## Related Abstractions",
            ""
        ]
        
        # Add related abstractions section
        for abstraction in answer_info.related_abstractions:
            content.extend([
                f"### {abstraction}",
                "",
                "**Key Points:**",
                ""
            ])
            
            # Add visual markers
            for marker in answer_info.visual_markers.get(abstraction, []):
                content.append(f"- {marker}")
            content.append("")
            
        # Add references section if any
        if answer_info.references:
            content.extend([
                "## Related Files",
                ""
            ])
            for ref in sorted(answer_info.references):
                content.append(f"- `{ref}`")
            content.append("")
            
        # Add tags section
        if answer_info.tags:
            content.extend([
                "## Tags",
                "",
                ", ".join(f"`{tag}`" for tag in answer_info.tags),
                ""
            ])
            
        return "\n".join(content)

    def _format_context(self, context: dict) -> str:
        """Format the context dictionary into a string for prompts."""
        sections = []
        
        if "abstractions" in context:
            sections.append("Abstractions:")
            for abstraction in context["abstractions"]:
                sections.append(f"- {abstraction['name']}: {abstraction['description']}")
        
        if "relationships" in context:
            sections.append("\nRelationships:")
            for rel in context["relationships"].get("details", []):
                sections.append(f"- {rel.get('label', 'Related to')}")
        
        if "files" in context:
            sections.append("\nRelevant Files:")
            for path, content in context["files"]:
                sections.append(f"- {path}")
        
        return "\n".join(sections)
