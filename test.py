from multiprocessing import Pool, cpu_count
import time

import multiprocessing as mp
import threading


class PersistentPool:

    def __init__(self, workers=None):
        self._mp_pool_lock = threading.Lock()
        self._pool = None
        if workers is None:
            self._pool_count = min(2, int(mp.cpu_count() * 0.50))
        else:
            self._pool_count = workers

    @property
    def mp_pool(self):
        with self._mp_pool_lock:
            if self._pool is None:
                self._pool = mp.Pool(self._pool_count)
        return self._pool

    def close(self):
        with self._mp_pool_lock:
            if self._pool:
                self._pool.close()
                self._pool = None

    def __del__(self):
        self.close()


def some_time(t):
    time.sleep(t)
    return t


if __name__ == "__main__":

    workers = cpu_count() - 2
    T = 0.000001

    scale = 10
    N = [[10] * 100, [1000]]
    N = [[nn * scale for nn in n] for n in N]

    for nn in N:
        t = time.time()
        for n in nn:
            with Pool(workers) as p:
                results = p.map(some_time, [T] * n)
        print("Not persistent:", nn, time.time() - t)

    PPool = PersistentPool(workers=workers)
    for nn in N:
        t = time.time()
        for n in nn:
            results = PPool.mp_pool.map(some_time, [T] * n)
        print("Not persistent:", nn, time.time() - t)
        PPool.close()
