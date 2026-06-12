import asyncio
from src.core.logic import _detect_triage_phase, _check_sufficiency, process_chat_message, initialize_components

async def main():
    initialize_components()
    user_input = """I have been feeling very tired lately, with frequent headaches and dizziness.
It started about 2 weeks ago and has been gradually getting worse.
The headache is about 7/10, I haven't been sleeping well and I'm very stressed at work.
No fainting or vision problems.
It's usually worse in the evening."""
    
    markers = _check_sufficiency([], user_input)
    print("Markers:", markers)
    
    phase = _detect_triage_phase([], "ROUTINE", user_input)
    print("Phase:", phase)
    
    response, risk, _ = await process_chat_message(user_input, [])
    print("\nRisk:", risk)
    print("Response:\n", response)

if __name__ == "__main__":
    asyncio.run(main())
