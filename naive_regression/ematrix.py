# Copyright 2023 Duality Technologies Inc.
#
# Authors: David Cousins, Ian Quah
# portions derived from the open source PALISADE python-demo repository and the PALISADE Open-Crannog repository
# those portions Authors: David Cousins, Yuriy Polyakov, Andrey Kim, Ian Quah
#
# Portions of GD and SGD derived from
# https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/

import copy
import random
import unittest
from collections import deque
import numpy as np

import openfhe

import operator
from typing import Union, List, Tuple, Dict
from enum import Enum
from typing import Optional

class EnumMathOp(Enum):
    ADD = 1
    SUB = 2
    MUL = 3


Numeric = Union[int, float]
NumericVector = List[Numeric]
CipherVector = openfhe.Ciphertext
AbstractVector = Union[NumericVector, CipherVector]

__version__ = "1.0"


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class EMatrixError(Exception):
    """ An exception class for EMatrix """
    pass


class EMatrix(object):
    """A simple Python encrypted matrix class with
    basic operations and operator overloading"""
    cc: openfhe.CryptoContext
    public_key: openfhe.PublicKey
    private_key: Optional[openfhe.PrivateKey]
    rot_k: int

    def __init__(self, m: int, n: int, init: bool = True, packing: str = "vertical", repeated: bool = False):
        """

        Args:
            m: the m dimension
            n: the n dimension
            init: whether to initialize the underlying plaintext data containers (assumes plaintext)
            packing: vertical packing or horizontal packing
            repeated: indicates if the matrix being created is a repeated one
        """
        if packing != "vertical":
            packing = "horizontal"
        self.packing = packing

        self.repeated = repeated
        self.encrypted = False

        self.rows = []
        self.cols = []

        self.m = m
        self.n = n
        if init:
            data = [[0] * self.getTrailingSize() for _ in range(self.getLeadingSize())]
            self.setData(data)

    def set_crypto(
            self,
            cc: openfhe.CryptoContext,
            rot_k: int,
            pub_k: openfhe.PublicKey,
            priv_k: Optional[openfhe.PrivateKey] = None
    ):
        self.__class__.cc = cc
        self.__class__.rot_k = rot_k
        self.__class__.public_key = pub_k
        self.__class__.private_key = priv_k

    def at(self, idx: int) -> "EMatrix":
        """
        Gets the vector at the index and returns it as an EMatrix,
            unlike __getitem__ which returns a NumericVector or CipherVector
        """
        result = EMatrix(self.m, self.n, packing=self.packing, repeated=False)
        result.setData([self[idx]])
        if self.isPackingHorizontal():
            result.m = 1
        else:
            result.n = 1
        result.encrypted = self.isEncrypted()
        return result

    def setData(self, data: List):
        """
        Sets all the underlying rows or cols to input data
        """
        # Assert it's a list of lists or a list of ciphertexts
        assert isinstance(data, list), "Data to be set must be a list"
        assert isinstance(data[0], list) or isinstance(data[0], openfhe.Ciphertext), "Data in list must be a list or ciphertext element"

        if self.isPackingHorizontal():
            self.rows = data
        else:
            self.cols = data

    def getData(self) -> List[AbstractVector]:
        """
        Gets all the underlying data (from either row or col depending on packing)

        NOTE: this is NOT necessary in all cases. In the case where we need an iterable,
            for _ in self should suffice but there are cases where we want to get the
            underlying data e.g when getting the list to print it or debug
        """
        if self.isPackingHorizontal():
            return self.rows
        return self.cols

    def __getitem__(self, idx: int) -> AbstractVector:
        if self.isPackingHorizontal():
            return self.rows[idx]
        return self.cols[idx]

    def __setitem__(self, idx: int, item: AbstractVector):
        if self.isPackingHorizontal():
            self.rows[idx] = item
        else:
            self.cols[idx] = item

    def __str__(self):
        if self.isEncrypted():
            s = " encrypted "
            return s + "\n" + self.packing + "\n"

        else:
            s = "\n".join(
                [" ".join([str(scalar) for scalar in vector]) for vector in self.getData()]
            )
            return s + "\n" + self.packing + "\n"

    def __repr__(self):
        ps = self.packing
        if self.isEncrypted():
            s = "encrypted"
        elif self.packing == "horizontal":
            s = str(self.rows)
        else:
            s = str(self.cols)

            rank = str(self.getRank())
            rep = 'EMatrix: packing: "%s" "%s", rank: "%s"' % (ps, s, rank)
            return rep

    def __len__(self):
        return self.getLeadingSize()
    def transposePacking(self) -> None:
        """
        Transpose the matrix by swapping the rows and cols.
        Swaps the rows and cols but does NOT transpose the underlying data
        """
        self.m, self.n = self.n, self.m
        temp = self.rows
        self.rows = self.cols
        self.cols = temp

        if self.isPackingHorizontal():
            self.packing = "vertical"
        else:
            self.packing = "horizontal"

    def getTransposePacking(self) -> "EMatrix":
        """Return a transpose of the matrix without
        modifying the matrix itself using TransposePacking"""

        m, n = self.n, self.m
        if self.isPackingHorizontal():
            mat = EMatrix(m, n, packing="vertical")
            mat.cols = self.rows
            mat.rows = []
        else:
            mat = EMatrix(m, n, packing="horizontal")
            mat.rows = self.cols
            mat.cols = []
        return mat

    def getRank(self) -> Tuple[int, int]:
        return self.m, self.n
    def getLeadingSize(self) -> int:
        if self.isPackingHorizontal():
            return self.m
        return self.n

    def getTrailingSize(self) -> int:
        if self.isPackingHorizontal():
            return self.n
        return self.m

    def getPacking(self) -> str:
        return self.packing

    def getLeadingSize(self):
        if self.isPackingHorizontal():
            return self.m
        return self.n

    def getTrailingSize(self):
        if self.isPackingHorizontal():
            return self.n
        return self.m

    def isPackingHorizontal(self) -> bool:
        return self.packing == "horizontal"

    def isPackingVertical(self) -> bool:
        return self.packing == "vertical"

    def isRepeated(self) -> bool:
        return self.repeated

    def isVector(self) -> bool:
        m, n = self.n, self.m
        return (m == 1) != (n == 1)  # equivalent of xor

    def isScalar(self) -> bool:
        return self.n == self.m == 1

    def isEncrypted(self) -> bool:
        return self.encrypted

    def isNotEncrypted(self) -> bool:
        return not self.encrypted

    def setEncrypted(self) -> None:
        self.encrypted = True

    def clearEncrypted(self) -> None:
        self.encrypted = False

    def __eq__(self, mat: "EMatrix") -> bool:
        """ Test equality would requires decryption, use for testing only"""
        if self.isEncrypted():
            raise EMatrixError("cannot test equality on encrypted data")

        if self.isRepeated() != mat.isRepeated():
            raise EMatrixError("Trying to test == of matrices of different repetition!")

        if self.isPackingVertical() != mat.isPackingVertical():
            raise EMatrixError("Trying to test ==  matrices of different packing!")

        return self.getData() == mat.getData()

    @staticmethod
    def __repetition_packing_rank_check(m1: "EMatrix", m2: "EMatrix", msg: str) -> None:
        """
        Checks that m1 and m2 are valid given the operations.
        1) They have the same rank (shape)
        2) They are both the same repetition
        3) They are both the same packing
        """
        if m1.getRank() != m2.getRank():
            raise EMatrixError(f"Trying to {msg} matricies of varying rank!")
        if m1.isRepeated() != m2.isRepeated():
            raise EMatrixError(f"Trying to {msg} matricies of different repetition!")
        if m1.packing != m2.packing:
            raise EMatrixError(f"Trying to {msg} matricies of different packing!")

    def __op_em_with_em(self, m1: "EMatrix", m2: "EMatrix", op: EnumMathOp) -> "EMatrix":
        """
        Abstract matrix op to support addition and subtraction with another EMatrix
        """

        if op == EnumMathOp.SUB:
            op_name = "sub"
            pt_op = operator.sub
            if m1.isEncrypted():
                enc_op = self.cc.EvalSub
        elif op == EnumMathOp.ADD:
            op_name = "add"
            pt_op = operator.add
            if m1.isEncrypted():
                enc_op = self.cc.EvalAdd
        elif op == EnumMathOp.MUL:
            op_name = "hprod"
            pt_op = operator.mul
            enc_op = self.cc.EvalMult
        else:
            raise EMatrixError(f"Unrecognized input to abstract matrix op: {op}")

        self.__repetition_packing_rank_check(m1, m2, op_name)
        ret = EMatrix(m1.m, m1.n, packing=m1.packing, repeated=m1.repeated)
        if m1.isNotEncrypted():  # plaintext Op
            for i in range(self.getLeadingSize()):
                vec = [pt_op(item[0], item[1]) for item in zip(m1[i], m2[i])]
                ret[i] = vec
        else:
            for i in range(self.getLeadingSize()):
                if isinstance(m1[i], list):
                    v_m1 = self.cc.MakeCKKSPackedPlaintext(m1[i])
                else:
                    v_m1 = m1[i]
                if isinstance(m2[i], list):
                    v_m2 = self.cc.MakeCKKSPackedPlaintext(m2[i])
                else:
                    v_m2 = m2[i]
                ret[i] = enc_op(v_m1, v_m2)
            ret.setEncrypted()
        return ret

    def __op_em_with_scalar(self, mat: "EMatrix", scalar: Numeric, op: EnumMathOp) -> "EMatrix":
        """
        Abstract matrix op to support addition and subtraction with a scalar
        """
        if op == EnumMathOp.SUB:
            pt_op = operator.sub
            if mat.isEncrypted():
                enc_op = self.cc.EvalSub
        elif op == EnumMathOp.ADD:
            pt_op = operator.add
            if mat.isEncrypted():
                enc_op = self.cc.EvalAdd
        elif op == EnumMathOp.MUL:
            pt_op = operator.mul
            if mat.isEncrypted():
                enc_op = self.cc.EvalMult
        else:
            raise EMatrixError(f"Unrecognized op to abstractScalar: {op}")
        ret = EMatrix(mat.m, mat.n, packing=mat.packing, repeated=mat.repeated)
        if mat.isNotEncrypted():  # plaintext scalar multiplicaton
            for i in range(self.getLeadingSize()):
                vec = [pt_op(_vec, scalar) for _vec in mat[i]]
                ret[i] = vec
        else:
            for i in range(self.getLeadingSize()):
                vec = enc_op(mat[i], scalar)
                ret[i] = vec
            ret.setEncrypted()
        return ret

    def __add__(self, invar: Union["EMatrix", Numeric]) -> "EMatrix":
        """
        Add an ematrix with the invar which is either a numeric or an EMatrix
            and returns a new matrix

        This works for both encrypted and plaintext
        """
        if isinstance(invar, (int, float)):
            return self.__op_em_with_scalar(self, invar, EnumMathOp.ADD)
        elif isinstance(invar, EMatrix):
            return self.__op_em_with_em(self, invar, EnumMathOp.ADD)
        else:
            raise EMatrixError("call to __add__ with bad variable type")

    def __sub__(self, invar: Union["EMatrix", Numeric]) -> "EMatrix":
        """
        Sub an ematrix with the invar which is either a numeric or an EMatrix
            and returns a new matrix

        This works for both encrypted and plaintext
        """

        if isinstance(invar, (int, float)):
            return self.__op_em_with_scalar(self, invar, EnumMathOp.SUB)
        elif isinstance(invar, EMatrix):
            return self.__op_em_with_em(self, invar, EnumMathOp.SUB)
        else:
            raise EMatrixError("call to __sub__ with bad variable type")

    def hprod(self, mat: "EMatrix") -> "EMatrix":
        """returns a hadamard product  of this matrix and
        return the new matrix. Doesn't modify
        the current matrix
        mat can be encrypted or unencrypted
        they must both be same rank and packing"""
        self.printStats(False, "in hprod self")
        mat.printStats(False, "in hprod mat")

        if self.isNotEncrypted():
            if mat.isEncrypted():
                raise EMatrixError("hprod: self is not encrypted, but other is encrypted")

        return self.__op_em_with_em(self, mat, EnumMathOp.MUL)

    def __mul__(self, invar: Union["EMatrix", Numeric]) -> "EMatrix":
        """Multiply a matrix with this invar
        return the new matrix. Doesn't modify
        the current matrix
        invar is matrix or scalar"""

        # local function to multiply matricies
        def __mulMat(lmat: "EMatrix", mat: "EMatrix") -> "EMatrix":
            """Multiply a matrix with this matrix and
            return the new matrix. Doesn't modify
            the current matrix"""
            matm, matn = mat.getRank()

            if lmat.n != matm:
                raise EMatrixError("Matrices wrong rank cannot be multipled!")

            if lmat.isRepeated() != mat.isRepeated():
                raise EMatrixError(
                    "Trying to multiply matricies of different repetition!"
                )

            if lmat.packing == mat.packing:
                raise EMatrixError("Trying to multiply matricies of same packing!")

            if lmat.isNotEncrypted():  # plaintext subtract
                if lmat.packing == "horizontal":
                    mat_t = mat.getTransposePacking()
                    mulmat = EMatrix(lmat.m, matn, packing=lmat.packing, repeated=lmat.repeated)
                    for x in range(lmat.m):
                        for y in range(mat_t.m):
                            mulmat[x][y] = sum(
                                [
                                    item[0] * item[1]
                                    for item in zip(lmat.rows[x], mat_t[y])
                                ]
                            )
                else:
                    mulmat = EMatrix(lmat.m, mat.n, packing=lmat.packing, repeated=lmat.repeated)
                    ind = 0
                    for v_tup in lmat:
                        # Note, as we iterate, these are individual values, not tuples
                        for ind_to_row, h_val in enumerate(mat[ind]):
                            for i, el in enumerate(v_tup):
                                thing = el * h_val
                                mulmat[ind_to_row][i] += thing
                        ind += 1
            else:
                raise EMatrixError("Trying to multiply encrypted matricies")
            return mulmat

        if isinstance(invar, (int, float)):
            return self.__op_em_with_scalar(self, invar, EnumMathOp.MUL)
        elif isinstance(invar, EMatrix):
            return __mulMat(self, invar)
        else:
            raise EMatrixError("call to __mul__ with bad variable type")

    def __iadd__(self, invar: "EMatrix") -> "EMatrix":
        tempmat = self + invar
        if self.isPackingHorizontal():
            for i in range(self.m):
                self.rows[i] = tempmat.rows[i]
        else:
            for i in range(self.n):
                self.cols[i] = tempmat.cols[i]
        return self

    def __isub__(self, invar: "EMatrix") -> "EMatrix":
        tempmat = self - invar
        if self.isPackingHorizontal():
            self.rows = tempmat.rows[:]
        else:
            self.cols = tempmat.cols[:]
        return self

    def hprodSelf(self, mat: "EMatrix") -> "EMatrix":
        """Hprod a matrix to this matrix.
        This modifies the current matrix"""

        # Calls hprod()
        tempmat = self.hprod(mat)
        for i in range(self.getLeadingSize()):
            self[i] = tempmat[i]
        return self

    def __imul__(self, invar: Union["EMatrix", Numeric]) -> "EMatrix":
        tempmat = self * invar
        for i in range(self.getLeadingSize()):
            self[i] = tempmat[i]
        return self

    def __copy__(self):
        # this actually does a deep copy for the encrypted code
        # by copying the internal ciphertexts
        mat = EMatrix(self.m, self.n, packing=self.packing, repeated=self.repeated)
        mat.cols = self.cols
        mat.rows = self.rows
        if self.isEncrypted():
            mat.encrypted = self.encrypted
            # copied_data = [self.cc.Copy(vec) for vec in self.getData()]
            mat.setData([vec for vec in self.getData()])
        return mat

    def encryptSelf(self) -> "EMatrix":
        """ encrypts the matrix, this modifies the current matrix"""
        if self.isEncrypted():
            raise EMatrixError("Trying to encrypt an encrypted matrix")

        # for el in self.getData():
        #     self.cc.Encrypt(self.public_key, self.cc.MakeCKKSPackedPlaintext(el))
        encrypted_data = [self.cc.Encrypt(self.public_key, self.cc.MakeCKKSPackedPlaintext(vec)) for vec in self.getData()]
        self.setData(encrypted_data)
        self.setEncrypted()
        return self

    def encrypt(self) -> "EMatrix":
        """ returns an encrypted version of this matrix """
        if self.isEncrypted():
            raise EMatrixError("Trying to encrypt an encrypted matrix")
        ret = copy.copy(self)
        ret.encryptSelf()
        return ret

    def decryptSelf(self) -> "EMatrix":
        """decrypts the matrix, this modifies the current matrix
        note this requres a server call"""
        if self.isNotEncrypted():
            raise EMatrixError("Trying to decrypt a plaintext matrix")

        # print("Decrypting self")
        for i in range(self.getLeadingSize()):
            dec_vec: openfhe.Plaintext = self.cc.Decrypt(self[i], self.private_key)
            self[i] = dec_vec.GetRealPackedValue()[0: self.getTrailingSize()]
        self.clearEncrypted()
        return self

    def decrypt(self) -> "EMatrix":
        """returns  a decryption of this matrix
        note this requires a server call"""

        if self.isNotEncrypted():
            raise EMatrixError("Trying to decrypt a plaintext matrix")
        ret = copy.copy(self)
        ret.decryptSelf()
        return ret

    def recrypt_self(self) -> "EMatrix":
        """decrypts the matrix, this modifies the current matrix
        note this requres a server call"""
        if self.isNotEncrypted():
            # recrypt on unencrypted is noop
            return self

        for i in range(self.getLeadingSize()):
            dec_vec = self.cc.Decrypt(self[i], self.private_key)
            self[i] = self.cc.Encrypt(
                self.public_key,
                self.cc.MakeCKKSPackedPlaintext(dec_vec.GetRealPackedValue()[0: self.getTrailingSize()])
            )

        return self

    def recrypt(self) -> "EMatrix":
        """returns  a recryption of this matrix
        note this requires a server call"""
        ret = copy.copy(self)
        if self.isEncrypted():
            ret.decryptSelf()
            ret.encryptSelf()
        return ret

    def sum(self) -> "EMatrix":
        """returns a summation over the packing of the matrix
        which is a 1D matrix of the same packing"""
        ret = EMatrix(self.m, self.n, packing=self.packing, repeated=self.repeated)
        if self.isNotEncrypted():  # plaintext sum
            for i in range(self.getLeadingSize()):
                ret[i] = [sum(item for item in self[i])]
        else:
            for i in range(self.getLeadingSize()):
                ret[i] = self.cc.EvalSum(self[i], next_power_of_2(self.getTrailingSize()))
            ret.setEncrypted()

        if ret.isPackingHorizontal():
            ret.n = 1
            return ret
        ret.m = 1
        return ret

    def round(self, places: int = 5) -> "EMatrix":
        """ reounds off the matrix elements to places"""
        if self.isEncrypted():
            raise EMatrixError("Trying to roundSelf an encrypted matrix")
        ret = EMatrix(self.m, self.n, packing=self.packing, repeated=self.repeated)
        ret.setData(
            [([round(item, places) for item in vec]) for vec in self.getData()]
        )
        return ret

    def roundSelf(self, places: int = 5) -> "EMatrix":
        """ returns a rounded version of this matrix """
        if self.isEncrypted():
            raise EMatrixError("Trying to round an encrypted matrix")
        tempmat = self.round(places)
        for i in range(self.getLeadingSize()):
            self[i] = tempmat[i]
        return self

    def vecConv2Hrep(self, nrep: int) -> "EMatrix":
        (nrows, ncols) = self.getRank()
        if not (self.isVector() and self.isPackingVertical()):
            raise EMatrixError("vecConv2Hrep called on non vertically packed vector")
        if self.isEncrypted():
            ring_dim_ov_2 = int(self.cc.GetRingDimension() / 2)
            # build eVhrep_out, empty for now, and set as encrypted
            # even though columns are not encrypted. they will be
            # physically replaced with encrypted columns.
            eVhrep_out = EMatrix.makeZero(
                nrep, nrows, packing="vertical", repeated=True
            )
            eVhrep_out.encrypted = True
            zeros = [0] * nrows  # generate mask of zeroes
            # eVv_res = self.cc.Encrypt(zeros) #and encrypt it
            for i in range(nrows):
                mask = EMatrix.fromList(zeros, packing="vertical")
                mask[0][i] = 1  # generate mask with a 1 for this feature
                # mask out the single feature

                eVv_tmp = self.hprod(mask)
                # sum over ring_dim_ov_2 to replicate in the entire vector
                eVv_tmp[0] = self.cc.EvalSum(eVv_tmp[0], ring_dim_ov_2)
                # set nrows to nrep
                eVv_tmp.m = nrep
                eVv_tmp.debug(False, "copied masked point")
                # eVv_tmp now is filled with eVv_tmp[i]
                eVhrep_out[i] = eVv_tmp[0]
            return eVhrep_out

        else:
            to_rep = self[0]
            container = []
            for _ in range(nrep):
                container.append(to_rep)
            ret = EMatrix.fromList(container, packing="vertical", repeated=True)
            return ret

    # TODO: Test this
    def dot(self, invec: "EMatrix", outpacking: str) -> "EMatrix":
        """dot cases that work
        vec-vec (note output is rank 1,1 vector so out packing is easy
           same packing same rank both output packings
        some matrix-vector combinations used in regresssion
        not tested yet
           same packing opposite rank both output packings
        """

        def __enc_dot_eMv_eVV_to_eVv(eMv_x: "EMatrix", eVv_w: "EMatrix") -> "EMatrix":  # USED in grad calc
            # internal code to do dot of eMv with eVV yielding eVv
            # probably needs to be renamed eMv
            eVv_w.printStats(False, "eVv_w")
            nrow, ncol = eMv_x.getRank()
            eMv_x.printStats(False, "eMv_x")
            zeros = [[0]] * ncol  # makes a column vector length ncol
            eVv_res = EMatrix.fromList(zeros, packing="vertical")
            eVv_res.printStats(False, "eVv zeros")
            if eMv_x.isEncrypted():
                eVv_res.encryptSelf()

            # loop over matrix rows
            for i in range(ncol):
                x_tmp = eMv_x.at(i)
                x_tmp.printStats(False, "eMv_x[i]")
                eVv_wxi = eVv_w.hprod(x_tmp)
                eVv_wxi.printStats(False, "eVv_wxi")

                enc_sum = eVv_wxi.sum()
                enc_sum.vecResizeSelf(ncol)
                enc_sum.printStats(False, "enc_sum after resize")

                # enc_sum contains the output in slot 0
                # rotate it to slot i, mask it off and add it to the output
                if i > 0:
                    enc_sum.vecShiftSelf(i)
                    enc_sum.printStats(False, "enc_sum after shift")
                mask = EMatrix.fromList(zeros, packing="vertical")

                mask[0][i] = 1
                enc_sum = enc_sum.hprod(mask)
                eVv_res += enc_sum
                # eVv_res now contains eVv in slots 0..nrow-1
            return eVv_res

        def __enc_dot_eMv_eVhrep_to_eVv(eMv_X, eVhrep_w):
            # USED in yhat calc
            # internal code to do dot of eMv with eVhrep yielding eVv
            nrow, ncol = eMv_X.getRank()
            eVv_res = EMatrix.fromList([0] * nrow, "vertical")
            if eMv_X.isEncrypted():
                eVv_res.encryptSelf()
            # loop over matrix columns
            for i in range(ncol):
                x_tmp = eMv_X.at(i)
                x_tmp.debug(False, "x_tmp")
                w_tmp = eVhrep_w.at(i)
                w_tmp.debug(False, "w_tmp")
                eVv_xwi = x_tmp.hprod(w_tmp)
                eVv_res += eVv_xwi
            return eVv_res

        # begin selection code here
        if self.isVector() and invec.isVector():
            # print("self.isVector")
            if invec.isVector():
                # print("invec.isVector")
                # call equivalent of  enc_dot_evV
                # result is 1d vertical UNUSED in crannog.
                selflen = max(self.m, self.n)
                inveclen = max(invec.m, invec.n)
                if selflen != inveclen:
                    raise EMatrixError("dot two vectors of unequal length")
                if self.getPacking() == invec.getPacking():
                    eVv_bxi = self.hprod(invec)
                    eVv_sum = eVv_bxi.sum()
                    if eVv_sum.getPacking() == outpacking:
                        return eVv_sum
                    else:
                        return eVv_sum.getTransposePacking()
                else:
                    raise EMatrixError("dot different packing not done yet")

            else:
                raise EMatrixError("cannot dot vector with matrix")

        else:  # self is matrix
            if self.isPackingHorizontal():
                if invec.isVector() and invec.isPackingVertical():
                    if outpacking == "horizontal":
                        raise EMatrixError(
                            "dot: call enc_dot_eMv_eVV_to_eVh() not written yet"
                        )
                    else:
                        return __enc_dot_eMv_eVV_to_eVv(self, invec)
                else:  # invec is eVh
                    # never happens
                    raise EMatrixError("dot: self eMH dot eVh not written yet")
                    # never called
            else:  # self is eMv:
                # if invec is eVhrep:
                if invec.isRepeated() and invec.isPackingVertical():
                    return __enc_dot_eMv_eVhrep_to_eVv(self, invec)
                elif invec.isPackingVertical():
                    # should be called _eMv_
                    return __enc_dot_eMv_eVV_to_eVv(self, invec)
                else:
                    # never happens
                    raise EMatrixError("dot: self eMv dot other not written yet")

    def vecResizeSelf(self, newLength: int) -> None:
        """expands a vector to length newLength, entries are undefined"""
        if self.isEncrypted():
            # should eventually check for if newLength>ringsize
            # just reset length.
            if self.packing == "horizontal":
                self.n = newLength
            else:
                self.m = newLength
            return
        else:  # simply expand or contract the vector
            selflen = max(self.m, self.n)
            if newLength <= selflen:
                # shorten vector
                if self.packing == "horizontal":
                    for x in range(self.m):
                        row = self.rows[x]
                        self[x] = row[:newLength]
                else:
                    for x in range(self.n):
                        col = self.cols[x]
                        self[x] = col[:newLength]
            else:
                # pad out vector with zeros
                if self.packing == "horizontal":
                    for x in range(self.m):
                        row = self.rows[x]
                        for i in range(len(row), newLength):
                            row.append(0)
                        self[x] = row
                    self.n = newLength
                else:
                    for x in range(self.n):
                        col = self.cols[x]
                        for i in range(len(col), newLength):
                            col.append(0)
                        self[x] = col[:newLength]
                    self.m = newLength

    def vecShiftSelf(self, shift_by: int) -> None:
        # print("vecShiftSelf ",i)
        """shifts a vector by i indicies
        note, this is not exactly what happens in the encrypted world"""
        if not self.isVector():
            raise EMatrixError("vecShiftSelf called on non vector")
        if self.isEncrypted():
            for i in range(self.getLeadingSize()):
                vec = self.enc_rotate_eVv(self[i], shift_by)
                self[i] = vec
        else:
            for i in range(self.getLeadingSize()):
                vec_deq = deque(self[i])
                vec_deq.rotate(shift_by)
                self[i] = list(vec_deq)

    def debug(self, should_log: bool = True, message: str = "Debug: ") -> None:
        if should_log:
            if self.isEncrypted():
                print(message, " Encrypted: ", self.decrypt())
            else:
                print(message, " Plaintext: ", self)

    def printStats(self, should_log: bool = True, message: str = "Stats: ") -> None:
        if should_log:
            print(
                f"{message} {'Encrypted' if self.isEncrypted() else 'Plaintext'}: rank {self.getRank()} {self.getPacking()} repeated: {self.isRepeated()}")

    def toListofLists(self) -> List[List]:
        if self.isEncrypted():
            return []
        elif self.isRepeated():
            # we only want one element of the repeat
            return [[vec[0]] for vec in self.getData()]
        return [[item for item in vec] for vec in self.getData()]

    @classmethod
    def _makeEMatrix(cls, rows: List[List], packing: str, repeated: bool) -> "EMatrix":
        m = len(rows)
        n = len(rows[0])
        # Validity check
        if any([len(row) != n for row in rows[1:]]):
            raise EMatrixError("inconsistent row length")
        mat = EMatrix(m, n, init=False, packing=packing, repeated=repeated)
        if packing == "horizontal":
            mat.rows = rows
        else:
            cols = [list(x) for x in zip(*rows)]  # transpose 2D list
            mat.cols = cols
        if hasattr(cls, "cc"):
            mat.cc = cls.cc
            mat.rot_k = cls.rot_k
            mat.public_key = cls.public_key
            mat.private_key = cls.private_key
            mat.rot_k = cls.rot_k
        return mat

    @classmethod
    def makeRandom(
            cls, m: int, n: int, low: int = 0, high: int = 10, scale: int = 10,
            packing: str = "vertical", repeated: bool = False
    ) -> "EMatrix":
        """
        Make a random int matrix with elements in the range (low-high)
        """

        obj = EMatrix(m, n, init=False, packing=packing, repeated=repeated)
        container = []
        for i in range(obj.getLeadingSize()):
            container.append([random.randrange(low, high) / scale for _ in range(obj.getTrailingSize())])
        obj.setData(container)
        return obj

    @classmethod
    def makeZero(cls, m: int, n: int, packing: str = "vertical", repeated: bool = False) -> "EMatrix":
        """ Make a zero-matrix of rank (mxn) """
        rows = [[0] * n for _ in range(m)]
        # because we use fromList() we do not care about rows vs columns here
        return cls.fromList(rows, packing=packing, repeated=repeated)

    @classmethod
    def makeId(cls, m: int, packing: str = "vertical") -> "EMatrix":
        """ Make identity matrix of rank (mxm) """

        rows = [[0] * m for _ in range(m)]
        idx = 0

        for row in rows:
            row[idx] = 1
            idx += 1
        # we just use rows, because rows and cols are symmetrical in this case
        return cls.fromList(rows, packing=packing)

    @classmethod
    def fromList(cls, _list: List, packing: str = "vertical", repeated: bool = False) -> "EMatrix":
        """Create a matrix by directly passing a list
        of lists. a 1D list becomes a column vector"""

        # E.g: EMatrix.fromList([[1 2 3], [4,5,6], [7,8,9]])
        #      EMatrix.fromList([7,8,9])
        # NOTE: list of list is mapped to rows or columns within makeEMatrix.
        assert isinstance(_list, List), f"fromList only supports lists. Received: {type(_list)}"
        rows = _list[:]

        # print("rows type :", type(rows), rows)
        # print("rows[0] type :", type(rows[0]), rows[0])
        if type(rows[0]) is not list:
            # print('passed a 1D list, make it 2D column vector')
            newrows = []
            for element in rows:
                newrows.append([element])
            # print(newrows)
            rows = newrows
        return cls._makeEMatrix(rows, packing=packing, repeated=repeated)

    def enc_rotate_eVv(self, eVv_in: CipherVector, i: int) -> CipherVector:
        u = i % self.rot_k
        j = int((i - u) / self.rot_k)
        u = -u
        j = -j
        tot_rot = 0
        if u != 0:
            eVv_out = self.cc.EvalRotate(eVv_in, u)
            tot_rot += u
        else:
            eVv_out = eVv_in
        if j * self.rot_k != 0:
            eVv_out = self.cc.EvalRotate(eVv_out, j * self.rot_k)
        tot_rot += j * self.rot_k
        if tot_rot != -i:
            print("error: total rotation done: ", tot_rot, " i: ", i)
        return eVv_out
    @staticmethod
    def sigmoid(eVv_y: "EMatrix") -> "EMatrix":
        """
        Sigmoid function that supports EMatrices. This works for both encrypted and plaintext EMatrix
        """
        if eVv_y.isEncrypted():
            eVv_sig = eVv_y
            eVv_sig[0] = eVv_y.cc.EvalLogistic(eVv_y[0], -4, 4)
        else:
            _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
            eVv_sig = EMatrix.fromList(
                [_sigmoid(i) for i in eVv_y[0]], packing=eVv_y.getPacking()
            )

            eVv_sig.printStats(False, "eVv_sig")
        return eVv_sig

    def bootstrap_self(self):
        for i in range(self.getLeadingSize()):
            if openfhe.get_native_int() == 128:
                self[i] = self.cc.EvalBootstrap(self[i])
            else:
                self[i] = self.cc.EvalBootstrap(self[i], 2)


if __name__ == "__main__":
    unittest.main()
