"""
A convenience module for forking multiple processes and capturing stdout,
for use in code developed for:

Ian H. Bell and Eric W. Lemmon,  "Automatic fitting of binary interaction parameters for multi-fluid
Helmholtz-energy-explicit mixture models", 2016

No dependencies aside from standard libraries included in python

By Ian H. Bell, NIST (ian.bell@nist.gov)

LICENSE: public domain, but please reference paper
"""
from __future__ import print_function
import sys, traceback, time
from multiprocessing import Process, Pipe

class RedirectText2Pipe(object):
    """
    An text output redirector
    """
    def __init__(self, pipe_inlet, file_object = None, prefix = ''):
        self.pipe_inlet = pipe_inlet
        self.prefix = ''
        self.file_object = file_object
        
    def write(self, string):
        if string.strip():
            self.pipe_inlet.send(str(string))
            if self.file_object is not None:
                self.file_object.write(prefix + str(string))
            
    def flush(self):
        return None
        
class Guppy(Process):
    def __init__(self, pipe_results, target, *args, **kwargs):
        Process.__init__(self)
        self.pipe_stdio = kwargs.pop('pipe_stdio', None)
        self.pipe_results = pipe_results
        self.target = target
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        if self.pipe_stdio is not None:
            redir = RedirectText2Pipe(self.pipe_stdio)
            sys.stdout = redir
            sys.stderr = redir
        
        try:
            res = self.target(*self.args, **self.kwargs)
            self.pipe_results.send(res)
        except BaseException as BE:
            traceback.print_exc()
            
        self.done()
            
    def done(self):
        self.pipe_results.send('p'+str(self.pid)+' DONE')
            
class Spawner(object):
    def __init__(self, inputs, Nproc_max = 1):
        """ 
        Initialize the Spawner class that manages the processes that are spun off 
        """
        
        self.processes = []
        self.inputs = inputs
        self.Nproc_max = Nproc_max
        
    def add_process(self):
        """
        If an input is waiting in the queue and a slot has opened up, add the process
        """
        if self.inputs and len(self.processes) < self.Nproc_max:
            input = self.inputs.pop(0)
            p = {}
            p['pipe_mine'], p['pipe_theirs'] = Pipe()
            p['pipe_stdio_mine'], p['pipe_stdio_theirs'] = Pipe()
            kwargs = input.get('kwargs',{})
            kwargs['pipe_stdio'] = p['pipe_stdio_theirs']
            p['proc'] = Guppy(p['pipe_theirs'], input['target'], *input['args'], **kwargs)
            p['proc'].daemon = True
            p['proc'].start()
            self.processes.append(p)
            print(len(self.inputs), ' inputs remain')
        else:
            return
        
    def run(self):
        results = []
        
        while self.inputs or self.processes:
            self.add_process()
            for p in self.processes:
                if p['pipe_stdio_mine'].poll():
                    print('p', p['proc'].pid, '>', p['pipe_stdio_mine'].recv())
                if p['pipe_mine'].poll():
                    res = p['pipe_mine'].recv()
                    if res == 'p'+str(p['proc'].pid)+' DONE':
                        while p['pipe_stdio_mine'].poll():
                            print('p', p['proc'].pid, '>', p['pipe_stdio_mine'].recv())
                        p['proc'].join()
                        self.processes.pop(self.processes.index(p))
                        #print 'process w/ pid ' + str(p['proc'].pid) + ' is done'
                    else:
                        results.append(res)
            time.sleep(0.01) # This time keeps the main thread from locking up because 
                             # it is constantly trying to read the pipes
        return results
            
def f(x):
    return sum(x)
    
if __name__=='__main__':
    # Call the process directly without redirecting stdout
    p = {}
    p['pipe_mine'], p['pipe_theirs'] = Pipe()
    p['pipe_stdio_mine'], p['pipe_stdio_theirs'] = Pipe()
    g = Guppy(p['pipe_theirs'], f, [1,2,3])
    g.run()
    print(p['pipe_mine'].recv())
    
    # Call the spawner
    spawner = Spawner([dict(target = f, args = (range(1, n),)) for n in range(500)], Nproc_max = 6)
    print(spawner.run())