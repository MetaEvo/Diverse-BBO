import numpy as np


class GL_PSO():
    def __init__(self,problem,option):
        self.__pm=0.01
        self.__NP=100
        self.__nsel=10
        self.__w=0.7298
        self.__c1=1.49618
        self.__sg=7
        self.__rho=0.2
        self.__dim=problem['ndim_problem']
        
        self.__fes=0
        self.__exemplar_stag=np.zeros(self.__NP)
    
        self.__MaxFEs = option['max_function_evaluations']
        self.rng = np.random.default_rng(option['seed_rng'])
        self.problem = problem['fitness_function']
        self.is_bound = option.get('is_bound', False)
        self.log_points = option.get('log_points', 10000) 
        self.log_index = 0
        self.log_interval = self.__MaxFEs // self.log_points
        self.records = []

    def __exemplar_crossover(self):
        rand_index=self.rng.integers(low=0,high=self.__NP,size=(self.__NP,self.__dim))
        xs=self.__particles['pbest_position']
        rand_par=xs[rand_index,np.arange(self.__dim)[None,:]]
        rand_pbest_val=self.__particles['pbest'][rand_index]
        filter=rand_pbest_val<self.__particles['pbest'][:,None]
        r=self.rng.random((self.__NP,self.__dim))
        uniform_crossover=r*self.__particles['pbest_position']+(1-r)*self.__particles['gbest_position'][None,:]
        self.__new_exemplar=np.where(filter,rand_par,uniform_crossover)

    def __exemplar_mutation(self):
        rand_pos=self.rng.uniform(low=self.__lb,high=self.__ub,size=(self.__NP,self.__dim))
        self.__new_exemplar=np.where(self.rng.random((self.__NP,self.__dim))<self.__pm,rand_pos,self.__new_exemplar)
    
    def __exemplar_selection(self,problem,init=False):
        new_exemplar_cost=self.__get_costs(problem,self.__new_exemplar)
        if init:
            self.__exemplar=self.__new_exemplar
            self.__exemplar_cost=new_exemplar_cost
        else:
            suv_filter=new_exemplar_cost<self.__exemplar_cost
            self.__exemplar=np.where(suv_filter[:,None],self.__new_exemplar,self.__exemplar)
            self.__exemplar_stag=np.where(suv_filter,np.zeros_like(self.__exemplar_stag),self.__exemplar_stag+1)
            self.__exemplar_cost=np.where(suv_filter,new_exemplar_cost,self.__exemplar_cost)
        
        min_exemplar_cost=np.min(self.__exemplar_cost)
        
        self.__found_best=np.where(min_exemplar_cost<self.__found_best,min_exemplar_cost,self.__found_best)

    def __exemplar_tour_selection(self):
        rand_index=self.rng.integers(low=0,high=self.__NP,size=(self.__NP,self.__nsel))
        rand_exemplar=self.__exemplar[rand_index]
        rand_exemplar_cost=self.__exemplar_cost[rand_index]
        min_exemplar_index=np.argmin(rand_exemplar_cost,axis=-1)  # bs, ps
        selected_exemplar=rand_exemplar[range(self.__NP),min_exemplar_index]
        return selected_exemplar
    
    def __exemplar_update(self,problem,init):
        self.__exemplar_crossover()
        self.__exemplar_mutation()
        self.__exemplar_selection(problem,init)
        
        filter=self.__exemplar_stag>self.__sg
        if np.any(filter):
            self.__exemplar=np.where(filter[:,None],self.__exemplar_tour_selection(),self.__exemplar)
    
    def run_episode(self,problem):
        self.__init_population(problem)
        is_done=False
        while not is_done:
            is_done,info=self.__update(problem)
            # print('gbest:{}'.format(self.__particles['gbest_val']))
        return info
    def optimize(self):
        problem = self.problem
        self.__init_population(problem)
        is_done = False
        while not is_done:
            is_done,info=self.__update(problem)
        while len(self.records) < self.log_points + 1:
            self.log_index += 1
            self.records.append({'log_points': self.log_index,'best_y':self.__particles['gbest_val']})
        return {'best_so_far_y':self.__particles['gbest_val'],
                'log_points' : self.records
                }
    def __init_population(self,problem):
        self.__ub=problem.ub
        self.__lb=problem.lb
        self.__fes=0
        self.__exemplar_cost=1e+50
        
        rand_pos=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.__NP,self.__dim))
        self.__max_velocity=self.__rho*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity,high=self.__max_velocity,size=(self.__NP,self.__dim))
        c_cost = self.__get_costs(problem,rand_pos) # ps
        
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]
        self.__max_cost=np.min(c_cost)
        # print("rand_pos.shape:{}".format(rand_pos.shape))

        self.__particles={'current_position': rand_pos.copy(), #  ps, dim
                        'c_cost': c_cost.copy(), #  ps
                        'pbest_position': rand_pos.copy(), # ps, dim
                        'pbest': c_cost.copy(), #  ps
                        'gbest_position':gbest_position.copy(), # dim
                        'gbest_val':gbest_val,  # 1
                        'velocity': rand_vel.copy(), # ps,dim
                        'gbest_index':gbest_index # 1
                        }

        self.__found_best=self.__particles['gbest_val'].copy()
        self.__exemplar_update(problem,init=True)

        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]
        # 初始化的时候记录初始记录点消息
        self.records.append({'log_points': self.log_index,'best_y': gbest_val})
    def __get_costs(self,problem,position):
        ps=position.shape[0]
        self.__fes+=ps
        cost=problem(position)
        return cost

    def __update(self,problem):
        is_end=False
        
        rand=self.rng.random((self.__NP,self.__dim))
        new_velocity=self.__w*self.__particles['velocity']+self.__c1*rand*(self.__exemplar-self.__particles['current_position'])
        new_velocity=np.clip(new_velocity,-self.__max_velocity,self.__max_velocity)

        new_position=self.__particles['current_position']+new_velocity

        new_velocity=np.where(new_position>problem.ub,new_velocity*-0.5,new_velocity)
        new_velocity=np.where(new_position<problem.lb,new_velocity*-0.5,new_velocity)
        new_position=np.clip(new_position,problem.lb,problem.ub)

        new_cost=self.__get_costs(problem,new_position)

        filters = new_cost < self.__particles['pbest']
        # new_cbest_val,new_cbest_index=torch.min(new_cost,dim=1)
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        filters_best_val=new_cbest_val<self.__particles['gbest_val']
        # update particles
        new_particles = {'current_position': new_position, # bs, ps, dim
                            'c_cost': new_cost, # bs, ps
                            'pbest_position': np.where(np.expand_dims(filters,axis=-1),
                                                        new_position,
                                                        self.__particles['pbest_position']),
                            'pbest': np.where(filters,
                                                new_cost,
                                                self.__particles['pbest']),
                            'velocity': new_velocity,
                            'gbest_val':np.where(filters_best_val,
                                                    new_cbest_val,
                                                    self.__particles['gbest_val']),
                            'gbest_position':np.where(np.expand_dims(filters_best_val,axis=-1),
                                                        new_position[new_cbest_index],
                                                        self.__particles['gbest_position']),
                            'gbest_index':np.where(filters_best_val,new_cbest_index,self.__particles['gbest_index'])
                            }
        
        self.__particles=new_particles

        if self.__fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])
            self.records.append({'log_points': self.log_index,'best_y': self.__particles['gbest_val']})
        self.__exemplar_update(problem,init=False)

        self.__found_best=np.where(self.__particles['gbest_val']<self.__found_best,self.__particles['gbest_val'],self.__found_best)

        # if problem.optimum is None:
        is_end = self.__fes >= self.__MaxFEs
        # else:
            # is_end = self.__fes >= self.__MaxFEs or self.__particles['gbest_val'] <= 1e-8
        if is_end:
            if len(self.cost) >= self.log_points + 1:
                self.cost[-1] = self.__particles['gbest_val']
            else:
                self.cost.append(self.__particles['gbest_val'])
        return is_end, {'cost': self.cost, 'fes': self.__fes}