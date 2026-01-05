# await_concurrency_demo.py
#
# Demonstrates three cases side-by-side:
# 1) sequential await (no overlap)      -> ~2s total
# 2) concurrent gather (overlap)        -> ~1s total
# 3) accidental blocking time.sleep     -> ticker freezes; ~2s total
#
# Run: python await_concurrency_demo.py

import asyncio
import time

def ts() -> str:
    return f"{time.time():.2f}"

async def async_job(name: str, seconds: float) -> str:
    print(ts(), f"{name}: start")
    await asyncio.sleep(seconds)  # yields control to event loop
    print(ts(), f"{name}: end")
    return f"result-{name}"

def blocking_job(name: str, seconds: float) -> str:
    print(ts(), f"{name}: start (BLOCKING)")
    time.sleep(seconds)  # blocks the thread / event loop
    print(ts(), f"{name}: end   (BLOCKING)")
    return f"result-{name}"

async def ticker(label: str, period: float = 0.2) -> None:
    """Prints periodically so you can see whether the event loop is responsive."""
    try:
        while True:
            print(ts(), f"{label}: tick")
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        print(ts(), f"{label}: ticker cancelled")
        raise

async def demo_sequential() -> None:
    print("\n=== 1) Sequential awaits (no overlap) ===")
    t0 = time.time()
    r1 = await async_job("A", 1.0)
    r2 = await async_job("B", 1.0)
    print(ts(), "results:", r1, r2)
    print(ts(), f"total: {time.time() - t0:.2f}s")

async def demo_concurrent() -> None:
    print("\n=== 2) Concurrent gather (overlap) ===")
    t0 = time.time()
    r1, r2 = await asyncio.gather(
        async_job("A", 1.0),
        async_job("B", 1.0),
    )
    print(ts(), "results:", r1, r2)
    print(ts(), f"total: {time.time() - t0:.2f}s")

async def demo_blocking_inside_async() -> None:
    print("\n=== 3) Blocking call inside async (BAD) ===")
    t0 = time.time()

    tick_task = asyncio.create_task(ticker("T"))  # should keep ticking if loop is alive
    try:
        # These are synchronous blocking calls: they freeze the event loop.
        blocking_job("A", 1.0)
        blocking_job("B", 1.0)
    finally:
        tick_task.cancel()
        try:
            await tick_task
        except asyncio.CancelledError:
            pass

    print(ts(), f"total: {time.time() - t0:.2f}s")

async def main() -> None:
    await demo_sequential()
    await demo_concurrent()
    await demo_blocking_inside_async()

if __name__ == "__main__":
    asyncio.run(main())
