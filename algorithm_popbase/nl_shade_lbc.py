import numpy as np
import copy
import scipy.stats as stats


def test(x):
    return np.sum(np.isnan(x))


class NL_SHADE_LBC():
    def __init__(self,problem,option):
        self.__pb = 0.4  # rate of best individuals in mutation
        self.__pa = 0.5  # rate of selecting individual from archive
        self.__dim = problem['ndim_problem']  # dimension of problem
        self.__m = 1.5
        self.__p_iniF = 3.5
        self.__p_iniCr = 1.0
        self.__p_fin = 1.5
        self.__Nmax = 23 * self.__dim  # the upperbound of population size
        self.__Nmin = 4  # the lowerbound of population size
        self.__NP = self.__Nmax  # the population size
        self.__NA = self.__NP  # the size of archive(collection of replaced individuals)
        self.__H = 20 * self.__dim
        self.__cost = np.zeros(self.__NP)  # the cost of individuals
        self.__archive = np.array([])  # the archive(collection of replaced individuals)
        self.__MF = np.ones(self.__H) * 0.5  # the set of step length of DE
        self.__MCr = np.ones(self.__H) * 0.9  # the set of crossover rate of DE
        self.__k = 0  # the index of updating element in MF and MCr
        self.__MaxFEs = option['max_function_evaluations']
        self.__FEs = 0
        self.gbest = 1e15
        self.rng = np.random.default_rng(option['seed_rng'])
        self.problem = problem['fitness_function']
        self.is_bound = option.get('is_bound', False)
        self.log_points = option.get('log_points', 10000) 
        self.log_index = 0
        self.log_interval = self.__MaxFEs // self.log_points
        self.records = []

    def __evaluate(self, problem, u):
        if problem.optimum is None:
            cost = problem.eval(u)
        else:
            cost = problem.eval(u) - problem.optimum
        return cost

    # Binomial crossover
    def __Binomial(self, x, v, cr):
        NP, dim = x.shape
        jrand = self.rng.integers(dim, size=NP)
        u = np.where(self.rng.random((NP, dim)) < cr.repeat(dim).reshape(NP, dim), v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    # Exponential crossover
    def __Exponential(self, x, v, cr):
        NP, dim = x.shape
        Crs = cr.repeat(dim).reshape(NP, dim)
        u = copy.deepcopy(x)
        L = self.rng.integers(dim, size=(NP, 1)).repeat(dim).reshape(NP, dim)
        R = np.ones(NP) * dim
        rvs = self.rng.random((NP, dim))
        i = np.arange(dim).repeat(NP).reshape(dim, NP).transpose()
        rvs[rvs > Crs] = np.inf
        rvs[i <= L] = -np.inf
        k = np.where(rvs == np.inf)
        ki = np.stack(k).transpose()
        if ki.shape[0] > 0:
            k_ = np.concatenate((ki, ki[None, -1] + 1), 0)
            _k = np.concatenate((ki[None, 0] - 1, ki), 0)
            ind = ki[(k_[:, 0] != _k[:, 0]).reshape(-1, 1).repeat(2).reshape(-1, 2)[:-1]].reshape(-1, 2).transpose()
            R[ind[0]] = ind[1]

        R = R.repeat(dim).reshape(NP, dim)
        u[(i >= L) * (i < R)] = v[(i >= L) * (i < R)]
        return u

    # update pa according to cost changes
    def __update_Pa(self, fa, fp, na, NP):
        if na == 0 or fa == 0:
            self.__pa = 0.5
            return
        self.__pa = (fa / (na + 1e-15)) / ((fa / (na + 1e-15)) + (fp / (NP - na + 1e-15)))
        self.__pa = np.minimum(0.9, np.maximum(self.__pa, 0.1))

    def __mean_wL_Cr(self, df, s):
        if np.sum(df) > 0.:
            w = df / np.sum(df)
            pg = (self.__MaxFEs - self.__FEs) * (self.__p_iniCr - self.__p_fin) / self.__MaxFEs + self.__p_fin
            res = np.sum(w * (s ** pg)) / np.sum(w * (s ** (pg - self.__m)))
            return res
        else:
            return 0.9

    def __mean_wL_F(self, df, s):
        if np.sum(df) > 0.:
            w = df / np.sum(df)
            pg = (self.__MaxFEs - self.__FEs) * (self.__p_iniF - self.__p_fin) / self.__MaxFEs + self.__p_fin
            return np.sum(w * (s ** pg)) / np.sum(w * (s ** (pg - self.__m)))
        else:
            return 0.5

    def __update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.__mean_wL_F(df, SF)
            self.__MF[self.__k] = mean_wL
            mean_wL = self.__mean_wL_Cr(df, SCr)
            self.__MCr[self.__k] = mean_wL
            self.__k = (self.__k + 1) % self.__MF.shape[0]
        else:
            self.__MF[self.__k] = 0.5
            self.__MCr[self.__k] = 0.9

    def __choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self.__NP
        ind_r = self.rng.integers(0, self.__H, size=gs)  # index
        C_r = np.minimum(1, np.maximum(0, self.rng.normal(loc=self.__MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.__MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.__population[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[self.rng.integers(self.__archive.shape[0])] = self.__population[old_id]

    def __NLPSR(self):
        self.__sort()
        N = np.round(self.__Nmax + (self.__Nmin - self.__Nmax) * np.power(self.__FEs / self.__MaxFEs,
                                                                          1 - self.__FEs / self.__MaxFEs))
        A = int(max(N, self.__Nmin))
        N = int(N)
        if N < self.__NP:
            self.__NP = N
            self.__population = self.__population[:N]
            self.__cost = self.__cost[:N]
        if A < self.__archive.shape[0]:
            self.__NA = A
            self.__archive = self.__archive[:A]

    def __init_population(self, problem):
        self.__NP = 23 * self.__dim
        self.__population = self.rng.random((self.__NP, self.__dim)) * (problem.ub - problem.lb) + problem.lb
        self.__cost = self.__evaluate(problem, self.__population)
        self.__FEs = self.__NP
        self.__archive = np.array([])
        self.__MF = np.ones(self.__H) * 0.5
        self.__MCr = np.ones(self.__H) * 0.9
        self.__k = 0
        # 为了记录bsf的x和y
        # 同时也得记录x的信息
        self.best_so_far_x = self.__population[np.argmin(self.__cost)]
        self.gbest = np.min(self.__cost)
        # self.log_index = 1
        self.cost = [self.gbest]
        # 初始化的时候记录初始记录点消息
        self.records.append({'log_points': self.log_index,'best_y': self.gbest})

    # step method for ensemble, optimize population for a few times
    def __update(self,
                 problem,  # the problem instance
                 ):
        if self.__NA < self.__archive.shape[0]:
            self.__archive = self.__archive[:self.__NA]
        self.__pa = 0.5
        NP = self.__NP
        # check record point lest missing it
        self.__sort()
        # select crossover rate and step length
        Cr, F = self.__choose_F_Cr()
        Cr = np.sort(Cr)
        # initialize some record values
        fa = 0  # sum of cost improvement using archive
        fp = 0  # sum of cost improvement without archive
        df = np.array([])  # record of cost improvement of each individual
        pr = np.exp(-(np.arange(
            NP) + 1) / NP)  # calculate the rate of individuals at different positions being selected in others' mutation
        pr /= np.sum(pr)
        na = 0  # the number of archive usage
        # randomly select a crossover method for the population
        pb_upper = int(np.maximum(2, NP * self.__pb))  # the range of pbest selection
        pbs = self.rng.integers(pb_upper, size=NP)  # select pbest for all individual
        count = 0
        duplicate = np.where(pbs == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 1:
            pbs[duplicate] = self.rng.integers(NP, size=duplicate.shape[0])
            duplicate = np.where(pbs == np.arange(NP))[0]
            count += 1
        xpb = self.__population[pbs]
        r1 = self.rng.integers(NP, size=NP)
        count = 0
        duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.integers(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
            count += 1
        x1 = self.__population[r1]
        rvs = self.rng.random(NP)
        r2_pop = np.where(rvs >= self.__pa)[0]  # the indices of mutation with population
        r2_arc = np.where(rvs < self.__pa)[0]  # the indices of mutation with archive
        use_arc = np.zeros(NP, dtype=bool)  # a record for archive usage, used in parameter updating
        use_arc[r2_arc] = 1
        if self.__archive.shape[0] < 25:  # if the archive is empty, indices above are canceled
            r2_pop = np.arange(NP)
            r2_arc = np.array([], dtype=np.int32)
        r2 = self.rng.choice(np.arange(NP), size=r2_pop.shape[0], p=pr)
        count = 0
        duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.choice(np.arange(NP), size=duplicate.shape[0], p=pr)
            duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
            count += 1
        x2 = np.zeros((NP, self.__dim))
        # scatter indiv from population and archive into x2
        if r2_pop.shape[0] > 0:
            x2[r2_pop] = self.__population[r2]
        if r2_arc.shape[0] > 0:
            x2[r2_arc] = self.__archive[
                self.rng.integers(np.minimum(self.__archive.shape[0], self.__NA), size=r2_arc.shape[0])]
        Fs = F.repeat(self.__dim).reshape(NP, self.__dim)  # adjust shape for batch processing
        vs = self.__population + Fs * (xpb - self.__population) + Fs * (x1 - x2)
        # crossover rate for Binomial crossover has a different way for calculating
        Crb = np.zeros(NP)
        if self.__FEs + NP > self.__MaxFEs // 2:
            tmp_id = min(NP, self.__FEs + NP - self.__MaxFEs // 2)
            Crb[-tmp_id:] = 2 * ((self.__FEs + np.arange(tmp_id) + NP - tmp_id) / self.__MaxFEs - 0.5)

        usB = self.__Binomial(self.__population, vs, Crb)
        usE = self.__Exponential(self.__population, vs, Cr)
        us = usB
        CrossExponential = self.rng.random(NP) > 0.5
        CrossExponential = CrossExponential.repeat(self.__dim).reshape(NP, self.__dim)
        us[CrossExponential] = usE[CrossExponential]
        # reinitialize values exceed valid range
        # us = us * ((-100 <= us) * (us <= 100)) + ((us > 100) + (us < -100)) * (self.rng.random((NP, dim)) * 200 - 100)
        # us[us < problem.lb] = (us[us < problem.lb] + problem.lb) / 2
        # us[us > problem.ub] = (us[us > problem.ub] + problem.ub) / 2
        
        # TODO unify the values exceed valid range 
        if self.is_bound:
            us = np.clip(us,  problem.lb,  problem.ub)

        cost = self.__evaluate(problem, us)
        optim = np.where(cost < self.__cost)[0]  # the indices of indiv whose costs are better than parent
        for i in range(optim.shape[0]):
            self.__update_archive(i)
        SF = F[optim]
        SCr = Cr[optim]
        df = (self.__cost[optim] - cost[optim]) / (self.__cost[optim] + 1e-9)
        arc_usage = use_arc[optim]
        fp = np.sum(df[arc_usage])
        fa = np.sum(df[np.array(1 - arc_usage, dtype=bool)])
        na = np.sum(arc_usage)
        self.__population[optim] = us[optim]
        self.__cost[optim] = cost[optim]

        if np.min(cost) < self.gbest:
            self.gbest = np.min(cost)
            self.best_so_far_x = us[np.argmin(cost)]

        self.__FEs += NP
        # adaptively adjust parameters
        self.__pb = 0.2 + 0.1 * (self.__FEs / self.__MaxFEs)
        self.__NLPSR()
        self.__update_M_F_Cr(SF, SCr, df)
        self.__update_Pa(fa, fp, na, NP)
                
        
        if self.__FEs % self.log_interval == 0:
            self.log_index += 1
            self.cost.append(self.gbest)
            self.records.append({'log_points': self.log_index,'best_y': self.gbest})

        if problem.optimum is None:
            return False
        else:
            return self.gbest <= 1e-8
        
        
    def optimize(self):
        problem = self.problem
        self.__init_population(problem)
        while self.__FEs < self.__MaxFEs:
            is_done = self.__update(problem)
            if is_done:
                break
        while len(self.records) < self.log_points + 1:
            self.log_index += 1
            self.records.append({'log_points': self.log_index,'best_y': self.gbest})
        # return {'cost': self.cost, 'fes': self.__FEs}
        # print(f"NL_SHADE_LBC : best_so_far_y {self.gbest} & Evalution {self.__FEs}")
        return {
                'best_so_far_y':self.gbest,
                'log_points' : self.records
                }

    def run_episode(self):
        problem = self.problem
        self.__init_population(problem)
        while self.__FEs < self.__MaxFEs:
            is_done = self.__update(problem)
            if is_done:
                break
        # if len(self.cost) >= self.__n_logpoint + 1:
        #     self.cost[-1] = self.gbest
        # else:
        #     self.cost.append(self.gbest)
        # return {'cost': self.cost, 'fes': self.__FEs}
        return {'best_so_far_x':self.best_so_far_x,
                'best_so_far_y':self.gbest,
                }
    


