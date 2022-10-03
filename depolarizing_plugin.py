import random
import numpy as np
from itertools import product
from copy import deepcopy

from qat.comm.datamodel.ttypes import OpType
from qat.comm.exceptions.ttypes import PluginException, ErrorType
from qat.comm.shared.ttypes import ProcessingType
from qat.comm.datamodel.ttypes import Op
from qat.core.plugins import AbstractPlugin
from qat.core.util import extract_syntax
from qat.core import Batch
from qat.core import Result, BatchResult
from qat.core.wrappers.result import Sample
from qat.core.wrappers.result import aggregate_data
from utils_tuto import make_matrix

from qat.core.circuit_builder.matrix_util import np_to_circ, circ_to_np
from qat.comm.datamodel.ttypes import Matrix, ComplexNumber, Op, GateDefinition, GSyntax 

class DepolarizingPluginVec(AbstractPlugin):
    r"""
    When used to build a stack with a perfect QPU, it is equivalent to a noisy QPU with a depolarizing noise model.

    Here, we define the depolarizing noise model by its action on the density matrix:
    
    .. math::
        \mathcal{E}(\rho) = (1 - p) \rho + \frac{p}{4^{n_\mathrm{qbits}}-1}\sum_{k = 1}^{4^{n_\mathrm{qbits}}-1} P_k \rho P_k

    where :math:`\lbrace P_k, k = 0 \dots 4^{n_\mathrm{qbits}} \rbrace` denotes
    the set of all products of Pauli matrices (including the identity) for
    :math:`n_\mathrm{qubits}` qubits. By convention, :math:`P_0 = I_{2^{n_\mathrm{qbits}}}`.

    Args:
        prob_1qb (float, optional): 1-qbit depolarizing probability.
            Defaults to 0.0.
        prob_2qb (float, optional): 2-qbit depolarizing probability.
            Defaults to 0.0.
        n_samples (int, optional): number of stochastic samples.
            Defaults to 1000.
        seed (int, optional): seed for random number generator.
            Defaults to 1425.
        verbose (bool, optional): for verbose output. Defaults to False.
    """
    
    def __init__(self, prob_1qb=0.0, prob_2qb=0.0, verbose=False):
        self.prob_1qb = prob_1qb
        self.prob_2qb = prob_2qb
        self.verbose = verbose

        self.nbshots = []
        self.nbqbits = []
        self.qubits = []
        self.job_type = []
        self.job_observable = []
        
    def compile(self, batch, harware_specs):
#        if len(batch.jobs) != 1:
#            raise PluginException(code=ErrorType.INVALID_ARGS,
#                                  message="This plugin supports only single jobs"
#                                  ", got %s instead"%len(batch.jobs))

        self.nbshots = []
        self.nbqbits = []
        self.qubits = []
        self.job_type = []
        self.job_observable = []
        
        new_batch = []
        for job_ind, job in enumerate(batch.jobs):
            self.nbshots.append(job.nbshots)
            self.nbqbits.append(job.circuit.nbqbits)
            self.qubits.append(job.qubits)
            self.job_type.append(job.type)
            self.job_observable.append(job.observable)
            
            list_2qb_paulis = ["%s%s"%(p1, p2)
                               for p1, p2 in product(["I", "X", "Y", "Z"],
                                                     repeat=2) 
                                if p1 != 'I' or p2 !='I']
            
            pauli_mats = {"I": np.identity(2),
                          "X": np.array([[0, 1], [1, 0]]),
                          "Y": np.array([[0, -1j], [1j, 0]]),
                          "Z": np.array([[1, 0], [0, -1]])
                         }
            one_qb_depol_superop = (1-self.prob_1qb)*np.identity(4, dtype=np.complex128)
            for pauli in ["X", "Y", "Z"]:
                one_qb_depol_superop += self.prob_1qb/3 * np.kron(pauli_mats[pauli], np.conj(pauli_mats[pauli]))
            depol1_superop = np_to_circ(one_qb_depol_superop)
            
            two_qb_depol_superop = (1-self.prob_2qb)*np.identity(16, dtype=np.complex128)
            for pauli in list_2qb_paulis:
                pauli_mat2 = np.kron(pauli_mats[pauli[0]], pauli_mats[pauli[1]])
                two_qb_depol_superop += self.prob_2qb/15 * np.kron(pauli_mat2, np.conj(pauli_mat2))
            depol2_superop = np_to_circ(two_qb_depol_superop)
            
            def _get_fresh_key(gate_dic):
                i = 0
                for k in gate_dic.keys():
                    if k[0] == '_' and k[1:].isdigit():
                        i = max(int(k[1:]), i)
                return '_' + str(i + 1)
                        
            job_copy = deepcopy(job)
            job_copy.nbshots = 0
            job_copy.qubits = None
            job_copy.type = ProcessingType.SAMPLE
            job_copy.observable = None
            job_copy.circuit.nbqbits = 2*self.nbqbits[job_ind]
            job_copy.circuit.ops = []
            
            if self.prob_1qb > 0:
                gate_def = GateDefinition(arity=2,
                                          name="depol1",
                                          matrix=depol1_superop,
                                          syntax=GSyntax(name="depol1", parameters=[]))
                job_copy.circuit.gateDic["depol1"] = gate_def
            if self.prob_2qb > 0:
                gate_def = GateDefinition(arity=4,
                                          name="depol2",
                                          matrix=depol2_superop,
                                          syntax=GSyntax(name="depol2", parameters=[]))
                job_copy.circuit.gateDic["depol2"] = gate_def
            
            for op in job.circuit:

                if op.type != OpType.GATETYPE:
                    raise PluginException(code=ErrorType.ILLEGAL_GATES,
                                          message="This plugin supports operators of type GATETYPE,"
                                                  " got %s instead"%op.type)
                if len(op.qbits) > 2:
                    gdef = job_copy.circuit.gateDic[op.gate]
                    gname = extract_syntax(gdef, job_copy.circuit.gateDic)[0]
                    if gname != "STATE_PREPARATION":
                        raise PluginException(code=ErrorType.NBQBITS,
                                              message="This plugin supports only 1 and 2-qbit gates,"
                                                      " got a gate acting on qbits %s instead"%op.qbits)
                        
                gdef = job_copy.circuit.gateDic[op.gate]    # retrieving useful info.
                if not gdef.matrix:
                    # StatePreparation case
                    gname = extract_syntax(gdef, job.circuit.gateDic)[0]
                    if gname == "STATE_PREPARATION":
                        matrix = gdef.syntax.parameters[0].matrix_p
                        np_matrix = circ_to_np(matrix)
                        if np_matrix.shape != (2**job.circuit.nbqbits, 1):
                            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.ILLEGAL_GATES,
                                               modulename="qat.pylinalg",
                                               file="qat/pylinalg/simulator.py",
                                               line=103,
                                               message="Gate {} has wrong shape {}, should be {}!"\
                                               .format(gname, np_matrix.shape, (2**job.circuit.nbqbits, 1)))
                        norm = np.linalg.norm(np_matrix) 
                        if abs(norm - 1.0) > 1e-10:
                            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.ILLEGAL_GATES,
                                               modulename="qat.pylinalg",
                                               file="qat/pylinalg/simulator.py",
                                               line=103,
                                               message="State preparation should be normalized, got norm = {} instead!"\
                                               .format(norm))
                        new_mat = np.kron(np_matrix, np.conj(np_matrix))
                        gdef.syntax.parameters[0].matrix_p = np_to_circ(new_mat)
                        op.qbits += [qb + self.nbqbits[job_ind] for qb in op.qbits]
                        job_copy.circuit.ops.append(op)
                        
                        
                        continue

                # first add gate U applied to qb q
                job_copy.circuit.ops.append(op)
                
                # then add gate U^* applied to qb q+nqbits
                conj_op = deepcopy(op)
                conj_op.qbits = [qb+self.nbqbits[job_ind] for qb in op.qbits]
                mat = circ_to_np(job_copy.circuit.gateDic[conj_op.gate].matrix)
                
                if np.linalg.norm(np.conj(mat) - mat) > 1e-12:
                    # need to include new gate in gateDic
                    new_key = _get_fresh_key(job_copy.circuit.gateDic)
                    gate_def = GateDefinition(arity=len(op.qbits),
                                              name=new_key,
                                              matrix=np_to_circ(np.conj(mat)),
                                              syntax=GSyntax(name=new_key, parameters=[]))
                    job_copy.circuit.gateDic[new_key] = gate_def
                    conj_op.gate = new_key
                job_copy.circuit.ops.append(conj_op)
                    
                    
                # then add gate = sum_k E_k x E_k^* applied to qubits q, q', q+nqbits, q'+nqbits
                if len(op.qbits) == 1 and self.prob_1qb > 0:
                    depol1_op = Op(gate="depol1", qbits=[op.qbits[0], op.qbits[0]+self.nbqbits[job_ind]], type=0)
                    job_copy.circuit.ops.append(depol1_op)
                if len(op.qbits) == 2 and self.prob_2qb > 0:
                    depol2_op = Op(gate="depol2", qbits=[op.qbits[0], op.qbits[1],
                                                         op.qbits[0]+self.nbqbits[job_ind], op.qbits[1]+self.nbqbits[job_ind]],
                                   type=0)
                    job_copy.circuit.ops.append(depol2_op)

            new_batch.append(job_copy)
        return Batch(new_batch, meta_data=batch.meta_data)
    
    def post_process(self, batch_result):
        
        result_list = []
        for job_ind, result in enumerate(batch_result.results):
            rho_vec = np.zeros(4**self.nbqbits[job_ind], np.complex128)
            for sample in result:
                rho_vec[sample.state.int] = sample.amplitude
                #print(sample.state.int, sample.amplitude)
            rho_vec = rho_vec.reshape(2**self.nbqbits[job_ind], 2**self.nbqbits[job_ind])

            if self.verbose:
                print("rho = ", rho_vec)
                print("tr (rho)=", np.trace(rho_vec))
                
            if self.job_type[job_ind] == ProcessingType.SAMPLE or self.job_observable[job_ind] is None:

                # tracing out some qubits
                if self.qubits[job_ind] != list(range(self.nbqbits[job_ind])):
                    qubits_to_trace_out = list(range(self.nbqbits[job_ind]))
                    for qb in self.qubits[job_ind]:
                        qubits_to_trace_out.remove(qb)

                    def partial_trace(rho, indices):
                        """trace out 'indices' from matrix 'rho' """
                        nbqbits = int(np.log2(rho.shape[0]))
                        shape = [2 for _ in range(nbqbits*2)]
                        rho = rho.reshape(*shape)
                        for qb in reversed(indices):
                            rho = np.trace(rho, axis1=qb, axis2=qb+nbqbits)
                            nbqbits -= 1
                        return rho
                        
                    rho_vec = partial_trace(rho_vec, qubits_to_trace_out)

                probs = rho_vec.diagonal()
                res = Result()
                res.raw_data = []
                if self.nbshots[job_ind] == 0:

                    for int_state, val in enumerate(probs):
                        sample = Sample(state=int_state,
                                        probability=np.real(val))
                        res.raw_data.append(sample)
                    result_list.append(res) 
                else:
                    cumul = np.cumsum(probs)  # cumulative distribution function.

                    intprob_list = []  # return object
                    for _ in range(self.nbshots[job_ind]):
                        res_int = np.searchsorted(cumul, np.random.random())  # sampling
                        sample = Sample(state=res_int)
                        res.raw_data.append(sample)
                    res = aggregate_data(res)
                    result_list.append(res) 

            elif self.job_type[job_ind] == ProcessingType.OBSERVABLE:
                if self.nbshots[job_ind] == 0:
                    O_mat = make_matrix(self.job_observable[job_ind])
                    res = np.trace(np.dot(rho_vec, O_mat))
                    result_list.append(Result(value=res, error=None)) 
                else:
                    raise Exception("nbshots > 0 not yet implemented")

            else:
                raise Exception("Unknown job type")

        self.nbshots = []
        self.nbqbits = []
        self.qubits = []
        self.job_type = []
        self.job_observable = []

        return BatchResult(results=result_list, meta_data=batch_result.meta_data)
