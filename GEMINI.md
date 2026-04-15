# Gemini Instructions

## Primary Directive
**Before taking any action or running any command**, you MUST read `MEMORY.md`. It acts as the root of our knowledge graph and contains the current state of the repository, architectural decisions, and research.

## Memory System Rules
1. **Act like a Wiki:** Update `MEMORY.md` and related documents in the `memory/` directory with your findings, research, and the current state of the pipeline.
2. **Link Everything:** Use relative hyperlinks to connect related concepts (e.g., `[Pipeline Architecture](memory/pipeline/architecture.md)`).
3. **Always Update:** After completing a task or stage, update the relevant memory documents to reflect the new state. This ensures that Claude or any other agent picking up the project will have the exact same context.
4. **Information Compression:** Keep top-level files concise and link to deeper nested files for extensive logs or detailed research.

Failure to read and update `MEMORY.md` will result in context loss and divergence between agents.
