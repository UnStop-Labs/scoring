import asyncio


class AsyncBarrier:
    """Synchronise multiple asyncio tasks at a rendezvous point."""

    def __init__(self, parties: int):
        if parties <= 0:
            raise ValueError("parties must be > 0")
        self.parties = parties
        self.count = 0
        self.condition = asyncio.Condition()
        self.generation = 0

    async def wait(self, timeout: float = 60.0):
        async with self.condition:
            gen = self.generation
            self.count += 1
            if self.count == self.parties:
                self.generation += 1
                self.count = 0
                self.condition.notify_all()
            else:
                try:
                    await asyncio.wait_for(self._wait_gen(gen), timeout=timeout)
                except asyncio.TimeoutError:
                    self.generation += 1
                    self.count = 0
                    self.condition.notify_all()

    async def _wait_gen(self, gen: int):
        while gen == self.generation:
            await self.condition.wait()
