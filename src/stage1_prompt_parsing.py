import os
import json
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ParsedPrompt(BaseModel):
    character_description: str = Field(description="The core description of the character's body, clothing, and appearance without style tags.")
    rigid_object: Optional[str] = Field(description="Any rigid object the character is holding or wearing that should not deform (e.g., staff, sword, mask).", default=None)
    animation_type: Literal["idle", "walk", "attack"] = Field(description="The inferred type of animation based on stance or action in the prompt. Must be 'idle', 'walk', or 'attack'.")
    style_tags: List[str] = Field(description="Style or technical tags like 'low-poly', 'game-ready', 'mobile-optimized', 'semi-voxel'.")


# ---------------------------------------------------------------------------
# Prompt wrapper — enforces semi-voxel game-art style for all characters
# ---------------------------------------------------------------------------

_STYLE_PREFIX = (
    "Semi-voxel stylized game character — blocky rounded proportions, "
    "exaggerated but readable features, suitable for hybrid mobile/PC games: "
)
_STYLE_SUFFIX = (
    ". Semi-voxel art style (think Clash of Clans / Fortnite proportions), "
    "game-ready humanoid character in neutral T-pose, full body, "
    "optimised for real-time 3D rendering."
)

def wrap_prompt(prompt: str) -> str:
    """
    Wraps any character prompt with semi-voxel game-art context and
    standard character-generation framing for hybrid mobile/PC games.

    Applied before the LLM call (Stage 1) so the parsed description and
    style_tags are already art-directed. Stage 2 expands the style_tags
    further into concrete SDXL keywords via _build_image_prompt().
    """
    return _STYLE_PREFIX + prompt.strip(" .") + _STYLE_SUFFIX


def parse_prompt(prompt: str, mock: bool = False) -> ParsedPrompt:
    """Parses a natural language prompt into a structured JSON format."""
    if mock:
        # Simple heuristic fallback for testing without a running LLM
        anim = "idle"
        if "combat" in prompt.lower() or "attack" in prompt.lower():
            anim = "attack"
        elif "walk" in prompt.lower() or "run" in prompt.lower() or "march" in prompt.lower():
            anim = "walk"
            
        rigid = None
        if "holding a " in prompt.lower():
            start = prompt.lower().find("holding a ") + 10
            # Extract until comma or end of string
            end = prompt.find(",", start)
            if end == -1: end = len(prompt)
            rigid = prompt[start:end].strip()

        # Extract some common style tags; always include semi-voxel
        tags = ["semi-voxel"]
        for tag in ["game-ready", "low-poly", "stylized", "mobile-optimized"]:
            if tag in prompt.lower():
                tags.append(tag)

        # Basic stripping for character desc (mock only)
        desc = prompt
        for tag in tags:
            desc = desc.replace(tag, "").replace(", ,", ",")
            
        return ParsedPrompt(
            character_description=desc.strip(" ,.-"),
            rigid_object=rigid,
            animation_type=anim,
            style_tags=tags
        )
        
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct")
    
    schema_desc = ParsedPrompt.model_json_schema()
    
    system_prompt = f'''You are a 3D asset generation assistant. 
Extract the core character description, any rigid object (like weapons or staves), the required animation type (strictly 'idle', 'walk', or 'attack'), and style tags from the user's prompt.
You MUST output valid JSON matching the following JSON schema:
{json.dumps(schema_desc, indent=2)}
'''

    # Apply semi-voxel game-art wrapper before sending to LLM
    wrapped = wrap_prompt(prompt)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this prompt: {wrapped}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Received empty response from LLM.")
        
    try:
        data = json.loads(content)
        # Handle case where some local LLMs nest the output under a key if not properly tuned
        if "ParsedPrompt" in data and isinstance(data["ParsedPrompt"], dict):
            data = data["ParsedPrompt"]
        return ParsedPrompt(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output into ParsedPrompt: {content}") from e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Prompt Parsing")
    parser.add_argument("prompt", type=str, help="The natural language prompt")
    parser.add_argument("--mock", action="store_true", help="Use mock parser instead of LLM")
    parser.add_argument("--output", type=str, default="output/parsed_prompt.json", help="Path to save the JSON output")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    result = parse_prompt(args.prompt, mock=args.mock)
    
    with open(args.output, "w") as f:
        f.write(result.model_dump_json(indent=2))
        
    print(f"Successfully parsed prompt and saved to {args.output}")
    print(result.model_dump_json(indent=2))
