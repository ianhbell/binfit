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
import tempfile, os, json, random, shutil, sys, time, array

# Other packages
import pandas
import numpy as np
from deap import algorithms, base, creator, tools
from six.moves import xrange # for python 2/3 compatibility

# Use the REFPROP wrapper provided by the ctypes python wrapper
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    
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
    
    lib = df[mask].sort_values(by='fluid[0] (-)')
    assert(len(lib) > 0)
    
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
    
def generate_vle_error(vle, RP, vle_type = 'bubble'):
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
            z = [vle['x[0] (-)'].iloc[i], vle['x[1] (-)'].iloc[i]]
            Q = 0
        elif vle_type == 'dew':
            z = [vle['y[0] (-)'].iloc[i], vle['y[1] (-)'].iloc[i]]
            Q = 1
        else:
            raise ValueError()
            
        try:
            kq = 1
            o = RP.TQFLSHdll(vle['T (K90)'].iloc[i], Q, z, kq)
            pcalc[i] = o.P*1000
            if o.ierr > 100:
                raise ValueError(o.herr)
            #pprint(i, vle['T (K90)'].iloc[i], o.p)
        except ValueError as VE:
            #pprint(VE)
            pcalc[i] = 1e20*vle['p (Pa)'].iloc[i]
            pass
        
    # Percentage relative error
    r = (pgiven - pcalc)/pgiven*100
    return r, pcalc, pgiven

def pad_parameters(params, fit_bits, defaults = None):
    """ Pad out the parameters list with the default values """
    if defaults is None:
        defaults = [1,1,1,1,0]
    o = []
    assert(len(defaults)==5)
    j = 0
    for i in range(5):
        if fit_bits[i]:
            o.append(params[j])
            j += 1
        else:
            o.append(defaults[i])
    return o

def set_parameters(RP, parameters):

    # Get the current state
    icomp, jcomp = 1, 2
    hmodij, fij, hfmix, hfij, hbinp, hmxrul = RP.GETKTVdll(icomp, jcomp)
    # Set the parameter values in the array of coefficients
    fij[0:len(parameters)] = array.array('d', parameters)
    # Set the parameters
    o = RP.SETKTVdll(icomp, jcomp, hmodij, fij, hfmix)
    
def apply_betagamma(df, parameters, ofname, Nloops = 100, Npoints_selected = 10, fit_bits = None):
    
    fluids = [df['fluid[0] (-)'].iloc[0], df['fluid[1] (-)'].iloc[0]]
    library = []
    tic = time.time()

    # Pad out the parameters with default values as necessary
    assert(fit_bits is not None) # make sure fit_bits is provided as a keyword argument
    parameters = pad_parameters(parameters, fit_bits)
        
    runt1 = time.time()
    
    root = os.environ['RPPREFIX']
    RP = REFPROPFunctionLibrary(os.path.join(root + '/REFPRP64.dll'), 'dll')
    RP.SETPATHdll(root)
    o = RP.SETUPdll(2, '|'.join([f + '.FLD' for f in fluids]), 'HMX.BNC', 'DEF')
    if o.ierr > 100:
        raise ValueError(o.herr)

    set_parameters(RP, parameters)
    _betaT, _gammaT, _betaV, _gammaV, _Fij = parameters

    Nfail = 0
    
    data = []
    for dummy_counter in xrange(Nloops):
        # r is a vector of percentage relative errors in p
        r, pcalc, pgiven = generate_vle_error(random_subset(df, 
                                                            Npoints_selected), 
                                              RP, 
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
    me = {'err_Nbest': err_Nbest, 'err_Nbestmedian': err_Nbestmedian, 'err_okmean':err_okmean, 
          'err_okmedian': err_okmedian, 'err_okrsse':err_okrsse, 
          'betaT':_betaT, 'gammaT':_gammaT, 'betaV':_betaV, 'gammaV':_gammaV, 'Fij':_Fij,
          'loop_elapsed':loop_elapsed}
    pprint(me)
    library.append(me)
    
    if (time.time() - tic) > 7200:
        with open(ofname,'w') as fp:
            fp.write('TIMEOUT')
    
    if ofname:
        pandas.DataFrame(library).to_csv(ofname)

    toc = time.time()

    # The penalty factor - F = 10^(penalty_factor*fail_fraction)
    penalty_factor = 3

    # If no results are ok, the mean of an empty array is NAN, so just return a huge number
    if np.isnan(err_okmean):
        return 1e10
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

def deap_optimizer(df, prefix = '', gammaT0 = None, force_only_gammaT = False, fit_bits = None):
    """
    This function actually calls deap and does the optimization, arriving at 
    the optimal values for beta_T and gamma_T

    Parameters
    ==========
    df: pandas.DataFrame
        Contains the VLE data
    prefix: string 
        A path prefix included on any files written out at the end of this function
    force_only_gammaT : bool
        If True, override fit_bits and only fit gammaT
    fit_bits : iterable
        A 5-element iterable object, in each element, a True (fit this parameter) or 
        False (don't fit this parameter) for each of the fittable parameters
        betaT,gammaT,betaV,gammaV,Fij (in order).  For example [1,1,0,0,0] says fit 
        betaT and gammaT
    """
    tic = time.time()
    fluids = df['fluid[0] (-)'].iloc[0], df['fluid[1] (-)'].iloc[0]

    # weight is -1 because we want to minimize the error
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMax)

    x1 = list(set(df['x[0] (-)']))
    T = list(set(df['T (K90)']))
    if (len(x1) == 1 or len(T) == 1 or np.min(x1) > 0.99*np.max(x1) 
        or np.min(T) > 0.99*np.max(T)) or force_only_gammaT:
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
        err = apply_betagamma(df, parameters, '', Nloops = 100, Npoints_selected = 5, fit_bits = fit_bits)
        with open(prefix + '&'.join(fluids)+'-deap-logger.csv', 'a+') as fp:
            fp.write('{0:s},{1:g}\n'.format(str(parameters), err))
        return (err,)
        
    toolbox = base.Toolbox()
    # See:
    # http://deap.readthedocs.org/en/master/tutorials/basic/part1.html#a-funky-one
    
    toolbox.register("attr_beta", random.uniform, 0.85, 1.0/0.85)
    toolbox.register("attr_gamma", random.uniform, 0.5, 4)
    toolbox.register("attr_Fij", random.uniform, 0.5, 4)

    if gammaT0 is not None and len(gammaT0) == 2:
        toolbox.register("attr_gammaT", random.uniform, gammaT0[0], gammaT0[1])
    else:
        toolbox.register("attr_gammaT", random.uniform, 0.5, 4.0)

    if fit_bits is None:
        # Fit betaT, gammaT, but not betaV, gammaV, or Fij
        fit_bits = [1,1,0,0,0]

    if only_fit_gammaT:
        generators = (toolbox.attr_gammaT,)
    else:
        gens = []
        if fit_bits[0]:
            gens.append(toolbox.attr_beta)
        if fit_bits[1]:
            gens.append(toolbox.attr_gamma)
        if fit_bits[2]:
            gens.append(toolbox.attr_beta)
        if fit_bits[3]:
            gens.append(toolbox.attr_gamma)
        if fit_bits[4]:
            gens.append(toolbox.attr_Fij)
        generators = tuple(gens)

    # Create the correctly sized individual
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                         generators, n = 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax, df = df, prefix = prefix)
    if 'DeltaPenality' in dir(tools):
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 100000))

    # If two individuals mate, interpolate between them, allow for a bit of extrapolation
    toolbox.register("mate", tools.cxBlend, alpha = 0.3) 

    # 
    sigma = [0.01]*len(generators)

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
    
    outs = []
    for p in pop:
        pp = pad_parameters(list(p), fit_bits)
        kwargs = dict(betaT = pp[0], gammaT = pp[1], betaV = pp[2], gammaV = pp[3], Fij = pp[4],
                      err = p.fitness.values[0])
        outs.append(kwargs)
    df = pandas.DataFrame(outs)
    df = df.sort_values(by='err')
    df.to_csv(prefix + '&'.join(fluids)+'-deap-data.csv')
            
    with open(prefix + '&'.join(fluids)+'-deap-log.csv','w') as fp:
        fp.write(str(log))
    with open(prefix + '&'.join(fluids)+'-deap-elapsed.csv','w') as fp:
        fp.write('Elapsed time(s)\n{tt:g}'.format(tt = time.time() - tic))
    
if __name__=='__main__':
    lib = pandas.read_excel('VLE_data.xlsx')

    # Uncomment this line to change the root directory of your python installation
    # os.environ['RPPREFIX'] = r'D:\Code\REFPROP-cmake\build\10\Release\\'

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
                   args = (collect_binary(lib, ('DECANE','PROPANE')),),
                   )
             ]
    spawn = Spawner(inputs, 
                    Nproc_max = 2 # Maximum number of subprocesses to fork
                    )
    spawn.run()