import os
import posix

import numpy as np

import neat
from neat import PhysTree, PhysNode, MorphLoc

import neuron
from neuron import h
h.load_file("stdlib.hoc") # contains the lambda rule
h.nrn_load_dll(os.path.join(os.path.dirname(__file__),
                            'x86_64/.libs/libnrnmech.so')) # load all mechanisms
# mechanism_name_translation NEURON
mechname = {'L': 'pas',
            'h': 'Ih', 'h_HAY': 'Ih_HAY',
            'Na': 'INa', 'Na_p': 'INa_p', 'Na_Ta': 'INa_Ta', 'Na_Ta2': 'INa_Ta2',
            'K': 'IK', 'Klva': 'IKlva', 'KA': 'IKA', 'm': 'Im', 'Kpst': 'IKpst', 'Ktst': 'IKtst', 'Kv3_1': 'IKv3_1', 'KA_prox': 'IKA_prox', 'SK': 'ISK',
            'Ca_LVA': 'ICa_LVA', 'Ca_HVA': 'ICa_HVA',
            'ca': 'conc_ca'}


class NeuronSimNode(PhysNode):
    def __init__(self, index, p3d=None):
        super(NeuronSimNode, self).__init__(index, p3d)

    def _makeSection(self, factorlambda=1., pprint=False):
        compartment = neuron.h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if self.index == 1:
            compartment.diam = 2. * self.R # um (NEURON takes diam=2*r)
            compartment.L = 2. * self.R    # um (to get correct surface)
            compartment.nseg = 1
        else:
            compartment.diam = 2. * self.R  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L # section length [um]
            # set number of segments
            if type(factorlambda) == float:
                # nseg according to NEURON bookint
                compartment.nseg = int(((compartment.L / (0.1 * h.lambda_f(100.)) + 0.9) / 2.) * 2. + 1.) * int(factorlambda)
            else:
                compartment.nseg = factorlambda

        # set parameters
        compartment.cm = self.c_m # uF/cm^2
        compartment.Ra = self.r_a*1e6 # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.iteritems():
            if current[0] > 1e-10:
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.g = ' + str(current[0]) + '*1e-6') # uS/cm^2 --> S/cm^2
                    exec('seg.' + mechname[key] + '.e = ' + str(current[1])) # mV
        h.pop_section()

        if pprint:
            print(self)
            print('>>> compartment length = %.2f um'%compartment.L)
            print('>>> compartment diam = %.2f um'%compartment.diam)
            print('>>> compartment nseg = ' + str(compartment.nseg))

        return compartment

    def _makeShunt(self):
        if self.g_shunt > 1e-10:
            shunt = h.Shunt(compartment(1.))
            shunt[-1].g = self.g_shunt # uS
            shunt[-1].e = self.v_eq   # mV
            return shunt
        else:
            return None


class NeuronSimTree(PhysTree):
    def __init__(self, file_n=None, types=[1,3,4],
                       factor_lambda=1., t_calibrate=0., dt=0.025, v_eq=-75.):
        super(NeuronSimTree, self).__init__(file_n=file_n, types=types)
        # neuron storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        # simulation parameters
        self.dt = dt # ms
        self.t_calibrate = t_calibrate # ms
        self.factor_lambda = factor_lambda
        self.v_eq = v_eq # mV
        self.indstart = int(t_calibrate / dt)

    def createCorrespondingNode(self, node_index, p3d=None):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        return NeuronSimNode(node_index, p3d=p3d)

    def initModel(self, dt=0.025, t_calibrate=0., v_eq=-75., factor_lambda=1.,
                        pprint=False):
        self.t_calibrate = t_calibrate
        self.dt = dt
        self.indstart = int(self.t_calibrate / self.dt)
        self.v_eq = v_eq
        self.factor_lambda = factor_lambda
        # reset all storage
        self.deleteModel()
        # create the NEURON model
        self._createNeuronTree(pprint=pprint)

    def deleteModel(self):
        # reset all storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        self.storeLocs([{'node': 1, 'x': 0.}], 'rec locs')
        # self.storeLocs([{'node': 1, 'x': 0.}], 'rec locs')
        # delete all hoc objects
        # h('forall delete_section()')

    def _createNeuronTree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._makeSection(self.factor_lambda, pprint=pprint)
            # connect with parent section
            if not self.isRoot(node):
                compartment.connect(self.sections[node.parent_node.index], 1, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._makeShunt()
            if shunt is not None:
                self.shunts.append(shunt)
        # if pprint:
        #     print(h.topology())


    def addShunt(self, loc, g, e_r):
        loc = MorphLoc(loc, self)
        # create the shunt
        shunt = h.Shunt(self.sections[loc['node']](loc['x']))
        shunt.g = g
        shunt.e = e_r
        # store the shunt
        self.shunts.append(shunt)

    def addDoubleExpCurrent(self, loc, tau1, tau2):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.epsc_double_exp(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        # store the synapse
        self.syns.append(syn)

    def addExpSyn(self, loc, tau, e_r):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau = tau
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def addDoubleExpSynapse(self, loc, tau1, tau2, e_r):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.Exp2Syn(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def addNMDASynapse(self, loc, tau, tau_nmda, e_r=0., nmda_ratio=1.7):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau = tau
        syn.tau_NMDA = tau_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def addDoubleExpNMDASynapse(self, loc, tau1, tau2, tau1_nmda, tau2_nmda,
                                     e_r=0., nmda_ratio=1.7):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.double_exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.tau1_NMDA = tau1_nmda
        syn.tau2_NMDA = tau2_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def addIClamp(self, loc, amp, delay, dur):
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.IClamp(self.sections[loc['node']](loc['x']))
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.amp = amp # nA
        # store the iclamp
        self.iclamps.append(iclamp)

    def addSinClamp(self, loc, amp, delay, dur, bias, freq, phase):
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.SinClamp(self.sections[loc['node']](loc['x']))
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.pkamp = amp # nA
        iclamp.bias = bias # nA
        iclamp.freq = freq # Hz
        iclamp.phase = phase # rad
        # store the iclamp
        self.iclamps.append(iclamp)


    def addOUClamp(self, loc, tau, mean, stdev, delay, dur, seed=None):
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        if tau > 1e-9:
            iclamp = h.OUClamp(self.sections[loc['node']](loc['x']))
            iclamp.tau = tau
        else:
            iclamp = h.WNclamp(self.sections[loc['node']](loc['x']))
        iclamp.mean = mean # nA
        iclamp.stdev = stdev # nA
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addOUconductance(self, loc, tau, mean, stdev, e_r, delay, dur, seed=None):
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUConductance(self.sections[loc['node']](loc['x']))
        iclamp.tau = tau
        iclamp.mean = mean # uS
        iclamp.stdev = stdev # uS
        iclamp.e_r = e_r # mV
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addOUReversal(self, loc, tau, mean, stdev, g_val, delay, dur, seed=None):
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUReversal(self.sections[loc['node']](loc['x']))
        iclamp.tau = tau # ms
        iclamp.mean = mean # mV
        iclamp.stdev = stdev # mV
        iclamp.g = g_val # uS
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addVClamp(self, loc, e_c, dur):
        loc = MorphLoc(loc, self)
        # add the voltage clamp
        vclamp = h.SEClamp(self.sections[loc['node']](loc['x']))
        vclamp.rs = 0.01
        vclamp.dur1 = dur
        vclamp.amp1 = e_c
        # store the vclamp
        self.vclamps.append(vclamp)

    def addRecLoc(self, loc):
        self.addLoc(loc, 'rec locs')

    def setSpikeTrain(self, syn_index, syn_weight, spike_times):
        # add spiketrain
        spks = np.array(spike_times) + self.t_calibrate
        spks_vec = h.Vector(spks.tolist())
        vecstim = h.VecStim()
        vecstim.play(spks_vec)
        netcon = h.NetCon(vecstim, self.syns[syn_index], 0, self.dt, syn_weight)
        # store the objects
        self.vecs.append(spks_vec)
        self.vecstims.append(vecstim)
        self.netcons.append(netcon)

    def run(self, t_max, downsample=1,
            record_from_syns=False, record_from_iclamps=False, record_from_vclamps=False,
            pprint=False):
        # simulation time recorder
        res = {'t': h.Vector()}
        res['t'].record(h._ref_t)
        # voltage recorders
        res['v_m'] = []
        for loc in self.getLocs('rec locs'):
            res['v_m'].append(h.Vector())
            res['v_m'][-1].record(self.sections[loc['node']](loc['x'])._ref_v)
        # synapse current recorders
        if record_from_syns:
            res['i_syn'] = []
            for syn in self.syns:
                res['i_syn'].append(h.Vector())
                res['i_syn'][-1].record(syn._ref_i)
        # current clamp current recorders
        if record_from_iclamps:
            res['i_clamp'] = []
            for iclamp in self.iclamps:
                res['i_clamp'].append(h.Vector())
                res['i_clamp'][-1].record(iclamp._ref_i)
        # voltage clamp current recorders
        if record_from_vclamps:
            res['i_vclamp'] = []
            for vclamp in self.vclamps:
                res['i_vclamp'].append(h.Vector())
                res['i_vclamp'][-1].record(vclamp._ref_i)

        # initialize
        # neuron.celsius=37.
        h.finitialize(self.v_eq)
        h.dt = self.dt

        # simulate
        if pprint: print '>>> Simulating the NEURON model for ' + str(t_max) + ' ms. <<<'
        start = posix.times()[0]
        neuron.run(t_max + self.t_calibrate)
        stop = posix.times()[0]
        if pprint: print '>>> Elapsed time: ' + str(stop-start) + ' seconds. <<<'
        runtime = stop-start

        # cast recordings into numpy arrays
        res['t'] = np.array(res['t'])[self.indstart:][::downsample] - self.t_calibrate
        for key in 'v_m', 'i_syn', 'i_clamp', 'i_vclamp':
            if key in res and len(res[key]) > 0:
                res[key] = np.array([np.array(reslist)[self.indstart:][::downsample] \
                                     for reslist in res[key]])
                if key in ('i_syn', 'i_clamp', 'i_vclamp'):
                    res[key] *= -1.

        return res

    def calcImpedanceMatrix(self, locarg, i_amp=0.001, t_dur=100., pplot=False):
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        z_mat = np.zeros((len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                self.initModel(dt=self.dt, t_calibrate=self.t_calibrate,
                               v_eq=self.v_eq, factor_lambda=self.factor_lambda)
                self.addIClamp(loc0, i_amp, 0., t_dur)
                self.storeLocs([loc0, loc1], 'rec locs')
                # simulate
                res = self.run(t_dur)
                # voltage deflections
                # v_trans = res['v_m'][1][-int(1./self.dt)] - self[loc1['node']].e_eq
                v_trans = res['v_m'][1][-int(1./self.dt)] - res['v_m'][1][0]
                # compute impedances
                z_mat[ii, jj] = v_trans / i_amp
                if pplot:
                    import matplotlib.pyplot as pl
                    pl.figure()
                    pl.plot(res['t'], res['v_m'][1])
                    pl.show()

        return z_mat

    def calcImpedanceKernelMatrix(self, locarg, i_amp=0.001,
                                                dt_pulse=0.1, t_max=100.):
        tk = np.arange(0., t_max, self.dt)
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        zk_mat = np.zeros((len(tk), len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                loc1 = locs[jj]
                self.initModel(dt=self.dt, t_calibrate=self.t_calibrate,
                               v_eq=self.v_eq, factor_lambda=self.factor_lambda)
                self.addIClamp(loc0, i_amp, 0., dt_pulse)
                self.storeLocs([loc0, loc1], 'rec locs')
                # simulate
                res = self.run(t_max)
                # voltage deflections
                v_trans = res['v_m'][1][1:] - self[loc1['node']].e_eq
                # compute impedances
                zk_mat[:, ii, jj] = v_trans / (i_amp * dt_pulse)
        return tk, zk_mat


class NeuronCompartmentNode(NeuronSimNode):
    def __init__(self, index):
        super(NeuronCompartmentNode, self).__init__(index)

    def getChildNodes(self, skip_inds=[]):
        return super(NeuronCompartmentNode, self).getChildNodes(skip_inds=skip_inds)

    def _makeSection(self, pprint=False):
        compartment = neuron.h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if 'points_3d' in self.content:
            points = self.content['points_3d']
            h.pt3dadd(*points[0], sec=compartment)
            h.pt3dadd(*points[1], sec=compartment)
            h.pt3dadd(*points[2], sec=compartment)
            h.pt3dadd(*points[3], sec=compartment)
        else:
            compartment.diam = 2. * self.R  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L # section length [um]
        # set number of segments to one
        compartment.nseg = 1

        # set parameters
        compartment.cm = self.c_m # uF/cm^2
        compartment.Ra = self.r_a*1e6 # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.iteritems():
            if current[0] > 1e-10:
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.g = ' + str(current[0]) + '*1e-6') # uS/cm^2 --> S/cm^2
                    exec('seg.' + mechname[key] + '.e = ' + str(current[1])) # mV
        h.pop_section()

        if pprint:
            print(self)
            print('>>> compartment length = %.2f um'%compartment.L)
            print('>>> compartment diam = %.2f um'%compartment.diam)
            print('>>> compartment nseg = ' + str(compartment.nseg))

        return compartment


class NeuronCompartmentTree(NeuronSimTree):
    def __init__(self, t_calibrate=0., dt=0.025, v_eq=-75.):
        super(NeuronCompartmentTree, self).__init__(file_n=None, types=[1,3,4],
                        t_calibrate=t_calibrate, dt=dt, v_eq=v_eq)

    # redefinition of bunch of standard functions to not include skip inds by default
    def __getitem__(self, index, skip_inds=[]):
        return super(NeuronCompartmentTree, self).__getitem__(index, skip_inds=skip_inds)

    def getNodes(self, recompute_flag=0, skip_inds=[]):
        return super(NeuronCompartmentTree, self).getNodes(recompute_flag=recompute_flag, skip_inds=skip_inds)

    def __iter__(self, node=None, skip_inds=[]):
        return super(NeuronCompartmentTree, self).__iter__(node=node, skip_inds=skip_inds)

    def _findNode(self, node, index, skip_inds=[]):
        return super(NeuronCompartmentTree, self)._findNode(node, index, skip_inds=skip_inds)

    def _gatherNodes(self, node, node_list=[], skip_inds=[]):
        return super(NeuronCompartmentTree, self)._gatherNodes(node, node_list=node_list, skip_inds=skip_inds)

    def createCorrespondingNode(self, node_index):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        return NeuronCompartmentNode(node_index)

    def _createNeuronTree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._makeSection(pprint=pprint)
            # connect with parent section
            if not self.isRoot(node):
                compartment.connect(self.sections[node.parent_node.index], 0.5, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._makeShunt()
            if shunt is not None:
                self.shunts.append(shunt)


def createReducedModel(ctree, fake_c_m=1., fake_r_a=100.*1e-6, method=2):
    # calculate geometry that will lead to correct constants
    arg1, arg2 = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                                 factor_r_a=1e-6, delta=1e-10,
                                                 method=method)
    if method == 1:
        points = arg1; surfaces = arg2
        sim_tree = ctree.__copy__(new_tree=NeuronCompartmentTree())
        for ii, comp_node in enumerate(ctree):
            pts = points[ii]
            sim_node = sim_tree.__getitem__(comp_node.index, skip_inds=[])
            sim_node.setP3D(np.array(pts[0][:3]), (pts[0][3] + pts[-1][3]) / 2., 3)

        # fill the tree with the currents
        for ii, sim_node in enumerate(sim_tree):
            comp_node = ctree[ii]
            sim_node.currents = {chan: [g / surfaces[comp_node.index], e] \
                                         for chan, (g, e) in comp_node.currents.iteritems()}
            sim_node.c_m = fake_c_m
            sim_node.r_a = fake_r_a
            sim_node.content['points_3d'] = points[comp_node.index]
    elif method == 2:
        lengths = arg1 ; radii = arg2
        surfaces = 2. * np.pi * radii * lengths
        sim_tree = ctree.__copy__(new_tree=NeuronCompartmentTree())
        for ii, comp_node in enumerate(ctree):
            sim_node = sim_tree.__getitem__(comp_node.index, skip_inds=[])
            if sim_tree.isRoot(sim_node):
                sim_node.setP3D(np.array([0.,0.,0.]), radii[ii]*1e4, 1)
            else:
                sim_node.setP3D(np.array([sim_node.parent_node.xyz[0]+lengths[ii]*1e4, 0., 0.]),
                                 radii[ii]*1e4, 3)

        # fill the tree with the currents
        for ii, sim_node in enumerate(sim_tree):
            comp_node = ctree[ii]
            sim_node.currents = {chan: [g / surfaces[comp_node.index], e] \
                                         for chan, (g, e) in comp_node.currents.iteritems()}
            sim_node.c_m = fake_c_m
            sim_node.r_a = fake_r_a
            sim_node.R = radii[comp_node.index]*1e4    # convert to [um]
            sim_node.L = lengths[comp_node.index]*1e4  # convert to [um]
    return sim_tree


# def _setSWCIndices(morph_tree):
#     for ii, node in enumerate(morph_tree.__iter__(skip_inds=[])):
#         if morph_tree.isRoot(node):
#             node.index = 1
#         else:
#             node.index = ii+3



