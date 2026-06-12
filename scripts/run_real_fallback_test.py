import asyncio
import logging
import sys
import io
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('.'))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
load_dotenv()

# We configure logging to capture everything to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from src.core.logic import process_chat_message

async def main():
    load_dotenv()
    
    print("==================================================")
    print("REAL TEST 1: Routine Medical Query")
    print("==================================================")
    res, risk, sources = await process_chat_message("I have a severe headache and dizziness. It has been 2 days. No fainting.", [])
    print("\n--- Final User-Visible Response ---")
    print(res)
    print("Risk Level:", risk)
    print("Sources:", sources)
    print("\n\n")

    print("==================================================")
    print("REAL TEST 2: Emergency Bypass")
    print("==================================================")
    res, risk, sources = await process_chat_message("I have crushing chest pain radiating to my left arm.", [])
    print("\n--- Final User-Visible Response ---")
    print(res)
    print("Risk Level:", risk)
    print("\n\n")

    print("==================================================")
    print("REAL TEST 3: Arabic Query")
    print("==================================================")
    res, risk, sources = await process_chat_message("أعاني من صداع شديد ومستمر منذ يومين. لا يوجد إغماء.", [])
    print("\n--- Final User-Visible Response ---")
    print(res)
    print("Risk Level:", risk)

if __name__ == "__main__":
    asyncio.run(main())
