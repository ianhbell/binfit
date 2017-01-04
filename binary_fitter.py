"""
This code carries out the fitting of the binary interaction parameters 
as described in:

Ian H. Bell and Eric W. Lemmon,  "Automatic fitting of binary interaction 
parameters for multi-fluid Helmholtz-energy-explicit mixture models", 2016

By Ian H. Bell, NIST (ian.bell@nist.gov)

The fitting was carried out using a development version of CoolProp:

>>> import CoolProp
>>> CoolProp.__version__
u'5.1.3dev'
>>> CoolProp.__gitrevision__
u'2688b572ac7fc18de4a77c4b930b5bddde52eda0'

You will also need numpy, pandas, and deap.  numpy and pandas can be obtained in 
the Anaconda installer package, deap should be installed from source.

There is a sample set of data for n-decane + n-propane mixture in an Excel 
spreadsheet which will be used if this script is run.

LICENSE: public domain, but please reference paper
"""
from __future__ import print_function

# Standard python modules
import tempfile, os, json, random, shutil, sys, time

# Other packages
import pandas
import numpy as np
import CoolProp, CoolProp.CoolProp as CP
from deap import algorithms, base, creator, tools
from six.moves import xrange # for python 2/3 compatibility

# Set the REFPROP path (if needed, mostly on linux where REFPROP needs a helping hand)
#jj = json.loads(CP.get_config_as_json_string())
#jj['ALTERNATIVE_REFPROP_PATH'] = '/home/ihb/fitting/'
#jj = CP.set_config_as_json_string(json.dumps(jj))

# Open the template file for REFPROP
with open('HMX.BNC.template','r') as fp:
    template = fp.read()
    
def pprint(*args):
    """ 
    A pretty-printing function that yields nicer output when printing from a 
    forked process 
    """
    print(' '.join(str(a) for a in args))
        
def collect_binary(df, pair):
    """
    Return a subsetted DataFrame including only the data associated with the 
    given pair of fluids
    """
    
    mask = ((df['Ncomp (-)'] == 2) 
             & (df['fluid[0] (-)'].isin(pair)) 
             & (df['fluid[1] (-)'].isin(pair)) 
             & (df['x[0] (-)'] + df['x[1] (-)'] > 1e-10) 
             & (df['x[0] (-)'] > 1e-10) & (df['x[1] (-)'] > 1e-10) 
             & (~np.isnan(df['p (Pa)'])) & (np.isnan(df['y[0] (-)']) 
                | ((df['y[0] (-)'] > 0) & (df['y[0] (-)'] < 1))))
    
    lib = df[mask].sort('fluid[0] (-)')
    
    if len(set(lib['fluid[0] (-)'])) > 1:
        fluid = list(set(lib['fluid[0] (-)']))[1]
        
        fluid_mask = (lib['fluid[0] (-)']==fluid)
        
        # Columns with a [0] or [1] in them
        zeros = [key.replace('0','!@#$') for key in lib.keys() if '[0]' in key]
        ones = [key.replace('1','!@#$') for key in lib.keys() if '[1]' in key]
        assert(zeros==ones)
        
        for key in zeros:
            one = key.replace('!@#$','1')
            zero = key.replace('!@#$','0')
            
            new_one_chunk = list(lib[fluid_mask][zero])
            new_zero_chunk = list(lib[fluid_mask][one])
            lib.loc[fluid_mask, zero] = new_zero_chunk
            lib.loc[fluid_mask, one] = new_one_chunk
        assert(len(set(lib['fluid[0] (-)'])) == 1)
        return lib.copy()
    else:
        return lib.copy()
    
def random_subset(lib, N):
    """ 
    Get a random subset of the data in the dataframe 
    """
    #random.seed(0) # Turn off randomness
    indices = random.sample(xrange(len(lib)), min(N, len(lib)))
    return lib.iloc[indices].copy()

def write_hmx_bnc(vle, params):
    """
    Write the HMX.BNC file for REFPROP, including the binary interaction 
    parameters 
    """
    if isinstance(vle,list):
        fluids = vle
    else:
        fluid0 = set(vle['fluid[0] (-)'])
        fluid1 = set(vle['fluid[1] (-)'])
        assert(len(fluid0) == 1)
        assert(len(fluid1) == 1)
        fluids = fluid0.pop(), fluid1.pop()
    
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    
    # Inputs to be written into the HMX.BNC file
    CAS1 = CoolProp.CoolProp.get_fluid_param_string('REFPROP::'+fluids[0],'CAS')
    CAS2 = CoolProp.CoolProp.get_fluid_param_string('REFPROP::'+fluids[1],'CAS')
    inputs = dict(  model = 'KWG',
                    Name1 = fluids[0],
                    CAS1 = CAS1,
                    Name2 = fluids[1],
                    CAS2 = CAS2,
                    betaT = float(params[0]),
                    gammaT = float(params[1]),
                    betaV = params[2],
                    gammaV = params[3],
                    Fij = params[4]
                )
                
    hmx_bnc_path = os.path.join(tmpdir, 'HMX.BNC')
    
    # Write the custom HMX.BNC file
    with open(hmx_bnc_path, 'w') as fp:
        fp.write(template.format(**inputs))
    
    # Tell CoolProp to use this new file for REFPROP
    jj = json.loads(CP.get_config_as_json_string())
    jj['ALTERNATIVE_REFPROP_HMX_BNC_PATH'] = hmx_bnc_path
    jj = CP.set_config_as_json_string(json.dumps(jj))
    
    return tmpdir
    
def generate_vle_error(vle, HEOS, vle_type = 'bubble'):
    """
    Call the VLE routines to generate the bubble pressure, 
    as well as the error in the bubble pressure as compared
    with the experimental data.
    """
    
    N = len(vle)
    pcalc = np.zeros((N,1))
    pgiven = np.array(vle['p (Pa)']).reshape((N,1))

    for i in xrange(N):
        if vle_type == 'bubble':
            HEOS.set_mole_fractions([vle['x[0] (-)'].iloc[i], 
                                     vle['x[1] (-)'].iloc[i]])
            Q = 0
        elif vle_type == 'dew':
            HEOS.set_mole_fractions([vle['y[0] (-)'].iloc[i], 
                                     vle['y[1] (-)'].iloc[i]])
            Q = 1
        else:
            raise ValueError()
            
        try:
            HEOS.update(CoolProp.QT_INPUTS, Q, vle['T (K90)'].iloc[i])
            pcalc[i] = HEOS.p()
            #pprint(i, vle['T (K90)'].iloc[i], HEOS.p())
        except ValueError as VE:
            #pprint(VE)
            pcalc[i] = 1e20*vle['p (Pa)'].iloc[i]
            pass
        
    # Percentage relative error
    r = (pgiven - pcalc)/pgiven*100
    return r, pcalc, pgiven
    
def apply_betaTgammaT(df, betaT, gammaT, ofname, Nloops = 100, Npoints_selected = 10):
    
    fluids = [df['fluid[0] (-)'].iloc[0], df['fluid[1] (-)'].iloc[0]]
    library = []
    tic = time.time()
    
    for j, (_betaT, _gammaT) in enumerate(zip(betaT, gammaT)):
        
        runt1 = time.time()
        tmpdir = write_hmx_bnc(df, [_betaT, _gammaT, 1, 1, 0])
        
        HEOS = CoolProp.AbstractState("REFPROP", "&".join(fluids))

        Nfail = 0
        
        data = []
        for dummy_counter in xrange(Nloops):
            # r is a vector of percentage relative errors in p
            r, pcalc, pgiven = generate_vle_error(random_subset(df, 
                                                                Npoints_selected), 
                                                  HEOS, 
                                                  'bubble')
            Nfail += sum(np.abs(r) > 1e6)
            err = np.sqrt(np.sum(r**2)) 
            data.append(err)
        # Fraction (0,1) of the runs that do not succeed and error out
        # We want to be able to "correct" the objective function by applying 
        # a penalty function if runs fail
        fail_fraction = float(Nfail)/float(Nloops*Npoints_selected)
        # pprint('fail_fraction:',fail_fraction)
        
        data = np.array(sorted(data))
        err_Nbest = np.mean(data[0:5])
        err_Nbestmedian = np.median(data[0:5])
        
        err_ok = data[(data < 1e6) & (~np.isnan(data))]
        if np.size(err_ok) > 0:
            err_okmean = np.mean(err_ok)
            err_okmedian = np.median(err_ok)
            err_okrsse = np.sqrt(np.sum(err_ok**2)/len(err_ok))
        else:
            err_okmean = 1e99
            err_okmedian = 1e99
            err_okrsse = 1e99
        
        loop_elapsed = time.time() - runt1
        pprint('fluid1, fluid2, betaT, gammaT, err_okmean, err_okrsse:', 
               fluids, float(_betaT), float(_gammaT), err_okmean, err_okrsse)
        library.append((err_Nbest, err_Nbestmedian, err_okmean, err_okmedian, 
                        err_okrsse, float(_betaT), float(_gammaT), 
                        loop_elapsed))
        
        if (time.time() - tic) > 7200:
            with open(ofname,'w') as fp:
                fp.write('TIMEOUT')
    
        # Clean up after ourselves
        shutil.rmtree(tmpdir)
        
    (err_Nbest, err_Nbestmedian, err_okmean, err_okmedian, 
        err_okrsse, betaT, gammaT, loop_elapsed) = zip(*library)
    
    if ofname:
        pandas.DataFrame(dict(err = err_Nbest, err_Nbestmedian = err_Nbestmedian, 
                              err_okmean = err_okmean, err_okmedian = err_okmedian, 
                              betaT = betaT, gammaT = gammaT, 
                              loop_elapsed = loop_elapsed)).to_csv(ofname)
    toc = time.time()

    # The penalty factor - F = 10^(penalty_factor*fail_fraction)
    penalty_factor = 3

    # If no results are ok, the mean of an empty array is NAN, so just return a huge number
    if np.isnan(err_okmean[0]):
        return 1e10
    else:
        if len(betaT) == 1:
            return err_okmean[0]*10**(penalty_factor*fail_fraction)
        else:
            return err_okmean*10**(penalty_factor*fail_fraction)
    
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=True, timeout = 3600):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.

    NOTE: This algorithm was taken from DEAP, and modified by Ian Bell
    in order to add a timeout
    """
    tic = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        pprint(logbook.stream)

    # Begin the generational process
    for gen in xrange(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            pprint(logbook.stream)

        if time.time() - tic > timeout:
            pprint('TIMEOUT:', time.time() - tic, 's elapsed')
            return population, logbook

    return population, logbook

def deap_optimizer(df, prefix = '', gammaT0 = None):
    """
    This function actually calls deap and does the optimization, arriving at 
    the optimal values for beta_T and gamma_T
    """
    tic = time.time()
    fluids = df['fluid[0] (-)'].iloc[0], df['fluid[1] (-)'].iloc[0]

    # weight is -1 because we want to minimize the error
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMax)

    x1 = list(set(df['x[0] (-)']))
    T = list(set(df['T (K90)']))
    if (len(x1) == 1 or len(T) == 1 or np.min(x1) > 0.99*np.max(x1) 
        or np.min(T) > 0.99*np.max(T)):
        only_fit_gammaT = True
    else:
        only_fit_gammaT = False
    pprint('number of unique T:', len(T), 'number of unique x:', len(x1), 
           'only_fit_gammaT:', only_fit_gammaT)
    
    def feasible(params):
        """
        Feasibility function for the individual. 
        Returns True if feasible, False otherwise.
        """
        if len(params) == 1 and 0.5 < params[0] < 4:
            return True
        elif 0.9 < params[0] < 1.1 and 0.5 < params[1] < 4:
            return True
        else:
            return False
    
    def evalOneMax(parameters, df = None, prefix = ''):
        if len(parameters) == 1:
            betaT = [1.0]
            gammaT = [parameters[0]]
        else:
            betaT = [parameters[0]]
            gammaT = [parameters[1]]
        err = apply_betaTgammaT(df, betaT, gammaT, '', Nloops = 100, Npoints_selected = 5)
        with open(prefix + '&'.join(fluids)+'-deap-logger.csv', 'a+') as fp:
            fp.write('{0:g},{1:g},{2:g}\n'.format(betaT[0], gammaT[0], err))
        return (err,)
        
    toolbox = base.Toolbox()
    # See:
    # http://deap.readthedocs.org/en/master/tutorials/basic/part1.html#a-funky-one
    
    toolbox.register("attr_betaT", random.uniform, 0.85, 1.0/0.85)
    if gammaT0 is not None and len(gammaT0) == 2:
        toolbox.register("attr_gammaT", random.uniform, gammaT0[0], gammaT0[1])
    else:
        toolbox.register("attr_gammaT", random.uniform, 0.5, 4.0)
    if only_fit_gammaT:
        toolbox.register("individual", tools.initCycle, creator.Individual, 
                         (toolbox.attr_gammaT,), n = 1)
    else:
        toolbox.register("individual", tools.initCycle, creator.Individual, 
                         (toolbox.attr_betaT, toolbox.attr_gammaT), n = 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax, df = df, prefix = prefix)
    if 'DeltaPenality' in dir(tools):
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 100000))
    # If two individuals mate, interpolate between them, allow for a bit of extrapolation
    toolbox.register("mate", tools.cxBlend, alpha = 0.3) 
    if only_fit_gammaT:
        sigma = [0.01]
    else:
        sigma = [0.01, 0.01]

    toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = sigma, indpb=1.0)
    # The greater the tournament size, the greater the selection pressure 
    toolbox.register("select", tools.selTournament, tournsize=5) 
    
    hof = tools.HallOfFame(50)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    pop = toolbox.population(n=150)
    pop, log = eaSimple(pop, 
                        toolbox, 
                        cxpb=0.5, # Crossover probability
                        mutpb=0.3, # Mutation probability
                        ngen=30, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True,
                        timeout = 7200)
    
    with open(prefix + '&'.join(fluids)+'-deap-data.csv','w') as fp:
        fp.write('betaT,gammaT,fitness\n')
        for p in pop:
            if only_fit_gammaT:
                kwargs = dict(betaT = 1.0, 
                              gammaT = p[0], 
                              err = p.fitness.values[0])
            else:
                kwargs = dict(betaT = p[0], 
                              gammaT = p[1], 
                              err = p.fitness.values[0])
            fp.write('{betaT:g},{gammaT:g},{err:g}\n'.format(**kwargs))
            
    with open(prefix + '&'.join(fluids)+'-deap-log.csv','w') as fp:
        fp.write(str(log))
    with open(prefix + '&'.join(fluids)+'-deap-elapsed.csv','w') as fp:
        fp.write('Elapsed time(s)\n{tt:g}'.format(tt = time.time() - tic))
    
if __name__=='__main__':
    lib = pandas.read_excel('VLE_data.xlsx')

    ################ SERIAL EVALUATION ###########
    ################ SERIAL EVALUATION ###########
    ################ SERIAL EVALUATION ###########

    # For serial evaluation, you can just call the deap_optimizer function
    deap_optimizer(collect_binary(lib, ('DECANE','PROPANE')))

    ################# PARALLEL EVALUATION ##################
    ################# PARALLEL EVALUATION ##################
    ################# PARALLEL EVALUATION ##################

    # Import the spawner module (built by IB)
    from spawn import Spawner

    inputs = [
              dict(target = deap_optimizer, 
                   args = (collect_binary(lib, ('DECANE','PROPANE'))),
                   )
             ]
    spawn = Spawner(inputs, 
                    Nproc_max = 2 # Maximum number of subprocesses to fork
                    )
    spawn.run()