import asyncio
import sys
from dotenv import load_dotenv
from pathlib import Path

# Ensure .env in backend/ is loaded
BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / '.env')

# Make sure project root is on sys.path so `src` package is importable
sys.path.insert(0, str(BASE))

from src.db.mongo import ping_db

async def main():
    try:
        await ping_db()
        print("PING_OK")
    except Exception as e:
        print("PING_FAIL")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == '__main__':
    asyncio.run(main())
