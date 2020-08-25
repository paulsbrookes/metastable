cdef extern from "gsl/gsl_sf_result.h":
  ctypedef struct gsl_sf_result:
    double val
    double err


cdef extern from "gsl/gsl_sf_mathieu.h":
  int gsl_sf_mathieu_a_e(int order, double qq, gsl_sf_result * result);
  int gsl_sf_mathieu_b_e(int order, double qq, gsl_sf_result * result);
  int gsl_sf_mathieu_ce_e(int order, double qq, double zz, gsl_sf_result *result);
  int gsl_sf_mathieu_se_e(int order, double qq, double zz, gsl_sf_result *result);


def mathieu_a(int order, double qq):
    cdef gsl_sf_result result
    cdef int outcome
    outcome = gsl_sf_mathieu_a_e(order,qq,&result)
    return result.val


def mathieu_b(int order, double qq):
    cdef gsl_sf_result result
    cdef int outcome
    outcome = gsl_sf_mathieu_b_e(order,qq,&result)
    return result.val


def mathieu_ce(int order, double qq, double zz):
    cdef gsl_sf_result result
    cdef int outcome
    outcome = gsl_sf_mathieu_ce_e(order,qq,zz,&result)
    return result.val


def mathieu_se(int order, double qq, double zz):
    cdef gsl_sf_result result
    cdef int outcome
    outcome = gsl_sf_mathieu_se_e(order,qq,zz,&result)
    return result.val
