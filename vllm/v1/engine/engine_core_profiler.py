###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import gc
import gzip
import json
import os
import queue
import math
import threading
import time
from abc import ABCMeta

from contextlib import contextmanager
from typing import Any, List
import psutil
import torch
import uuid
import inspect
import functools
from vllm.logger import init_logger

logger = init_logger(__name__)

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()  # Thread safety

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class FileWriter(threading.Thread):

    def __init__(self, filename, event_queue):
        super().__init__()
        self.filename = filename
        self.event_queue = event_queue
        self.daemon = True
        self.timer_event = threading.Event()

    def _drain_event_queue(self):
        content = ''
        while True:
            try:
                element = self.event_queue.get_nowait()
                content += element
            except queue.Empty:
                break
        return content

    def run(self):
        # don't check the queue too often
        while not self.timer_event.wait(1):
            # Block and wait for the next item in the queue
            content = self.event_queue.get()
            # Collect any other items in the queue
            content += self._drain_event_queue()

            with open(self.filename, 'a') as outfile:
                outfile.write(content)


class EngineCoreProfiler(metaclass=SingletonMeta):
    profiling_trace_events: queue.Queue = queue.Queue()
    event_tid = {'counter': 1, 'external': 2, 'internal': 3, 'scheduler': 4}
    event_cache: List[Any] = []

    def __init__(self, vllm_instance_id = None):
        self.path = os.getenv('VLLM_TORCH_PROFILER_DIR', 'false').lower()
        self.enabled = self.path != 'false'
        self.pid = os.getpid()
        if self.enabled:
            self.vllm_instance_id = vllm_instance_id if vllm_instance_id is not None \
                else f"vllm-instance-{self.pid}-{str(uuid.uuid4().hex)}"
            msg = f'Profiler enabled for: {self.vllm_instance_id}'
            logger.info(msg)
            os.makedirs(self.path, exist_ok=True)
            self.filename = f'{self.path}/server_events_{self.vllm_instance_id}.json'
            # initialize the trace file (JSON Array Format)
            with open(self.filename, 'w') as outfile:
                outfile.write('[')
            file_writer = FileWriter(self.filename,
                                     self.profiling_trace_events)
            file_writer.start()

    def _dump_with_sep(self, entry):
        entry = json.dumps(entry) + ','
        self.profiling_trace_events.put(entry)

    def get_timestamp_us(self):
        return time.time() * 1000000.0

    def record_counter(self, ts, counter):
        if self.enabled:
            self._dump_with_sep({
                'pid': self.pid,
                'tid': self.event_tid['counter'],
                'ph': 'C',
                'name': 'utils',
                'ts': ts,
                'args': counter
            })

    def start(self, type, name, args=None):
        if self.enabled:
            ts = self.get_timestamp_us()
            if args is not None and 'counter' in args:
                self.record_counter(ts, args['counter'])
                del args['counter']
            event = {
                'pid': self.pid,
                'tid': self.event_tid[type],
                'ph': 'X',
                'name': name,
                'ts': ts,
                'dur': None,
                'args': args
            }
            self.event_cache.append(event)

    def end(self):
        if self.enabled:
            ts = self.get_timestamp_us()
            if not self.event_cache:
                logger.warning(
                    'Profiler: end() call does not have matching start() call. '
                    'Disabling profiler.')
                self.enabled = False
                return
            event = self.event_cache.pop()
            event['dur'] = ts - event['ts']
            self._dump_with_sep(event)


    @contextmanager
    def record_event(self, type, name, args=None):
        if self.enabled:
            self.start(type, name, args)
            yield
            self.end()
        else:
            yield


profiler = EngineCoreProfiler()
def profiler_wrapper(func):
    if inspect.isgeneratorfunction(func):
        @functools.wraps(func)
        def gen_wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            with profiler.record_event('internal', func.__qualname__):
                yield from gen
        return contextlib.contextmanager(gen_wrapper)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with profiler.record_event('internal', func.__qualname__):
            return func(*args, **kwargs)
    return wrapped



class ProfiledMeta(type):
    def __new__(cls, name, bases, namespace):
        for attr, val in namespace.items():
            # Skip non-callables or dunder methods
            if not callable(val) or attr.startswith("__"):
                continue
            # Skip classmethods and staticmethods (for now)
            if isinstance(val, (staticmethod, classmethod)):
                continue
            # Wrap instance methods only
            namespace[attr] = profiler_wrapper(val)
        return super().__new__(cls, name, bases, namespace)

class ProfiledMetaABC(ABCMeta):
    def __new__(cls, name, bases, namespace):
        def profiler_wrapper(func):
            if inspect.isgeneratorfunction(func):
                @functools.wraps(func)
                def gen_wrapper(*args, **kwargs):
                    gen = func(*args, **kwargs)
                    with profiler.record_event('scheduler', func.__qualname__):
                        yield from gen
                return contextlib.contextmanager(gen_wrapper)

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with profiler.record_event('scheduler', func.__qualname__):
                    return func(*args, **kwargs)
            return wrapped

        for attr, val in namespace.items():
            if not callable(val) or attr.startswith("__"):
                continue
            if isinstance(val, (staticmethod, classmethod)):
                continue  # Extend this if you want to handle them
            namespace[attr] = profiler_wrapper(val)

        return super().__new__(cls, name, bases, namespace)