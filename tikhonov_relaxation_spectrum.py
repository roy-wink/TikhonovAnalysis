# -----------------------------------------------------------------------------
#
# Python script for determining the relaxation modes in relaxation spectra
# Version 1.1, August 2024, tested in Python 3.9.18 with NumPy 1.26.4
#
# © Roy Wink, TU Eindhoven, 2024
#
# -----------------------------------------------------------------------------
#
# Highly based on the MATLAB script of Per Christian Hansen:
#     P.C. Hansen, Regularization Tools:
#     A Matlab package for analysis and solution of discrete ill-posed problems
#     Numerical Algorithms 1994, 6 (1), 1–35.
#
# And the discr MATLAB function by Kontogiorgos et al:
#     V. Kontogiorgos, B. Jiang, S. Kasapis,
#     Numerical Computation of Relaxation Spectra from Mechanical Measurements
#     in Biopolymers. Food Research International 2009, 42 (1), 130–136.
#
# -----------------------------------------------------------------------------


# ----- ENTER SETTINGS HERE ---------------------------------------------------

from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
data_input = 'path\\input_data.txt'
data_output = 'path\\output_data.txt'

export_lcurve = True

# -----------------------------------------------------------------------------

# import matlab.engine


def discr(low, high, times):
    """
    Generate the s-space representing relaxation times and construct matrix A.

    Args:
        low (float): The lowest limit of the relaxation times in the s-space.
        high (float): The highest limit of the relaxation times in the s-space.
        times (numpy.ndarray): The time vector measured during the experiment.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:
            - A (numpy.ndarray): The constructed matrix A.
            - sp (numpy.ndarray): The s-space vector representing the
                                  (logaritmic) range of relaxation times.

    Example:
        A, sp = discr(1e-4, 1e6, data[:, 1])
    """

    # create logspace
    space = np.logspace(np.log10(low), np.log10(high), len(times))

    # transform the logspace and times to column vector
    space = space.reshape(-1, 1)
    times = times.reshape(-1, 1)

    # create mesh grids
    space_mesh, times_mesh = np.meshgrid(space, times)

    # generate matrix A by element-wise division
    # this forms the matrix space for singular value decomposition
    A = np.exp(np.divide(-times_mesh, space_mesh))

    # visualise matrix A
    # plt.imshow(A, interpolation='none')
    # plt.show()

    return A, space


def csvd(A, validation=False):
    """
    Perform compact singular value decomposition on matrix A.

    Args:
        A (numpy.ndarray): The input matrix for singular value decomposition.
        validation (bool, optional): Flag indicating whether validation steps
                                     should be performed. Defaults to False.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            A tuple containing:
            - U (numpy.ndarray): The left singular vectors,
                                 with shape (m, min(m, n)).
            - s (numpy.ndarray): The singular values as a diagonal matrix,
                                 with shape (min(m, n), 1).
            - V (numpy.ndarray): The right singular vectors,
                                 with shape (n, min(m, n)).

    Example:
        U, s, V = csvd(A)
    """

    U, S, V = np.linalg.svd(A, full_matrices=False)
    sd = np.diag(S)
    Vt = np.transpose(V)

    # validate that U * S * V equals A
    if validation:
        reconstructed_A = U @ sd @ V

        # check if the reconstructed matrix is close to the original A
        if np.array_equal(A.shape, reconstructed_A.shape) and np.allclose(A, reconstructed_A, rtol=1e-10):
            print('Validation: U * S * V equals A')
        else:
            print('Validation failed: U * S * V does not equal A')

    return U, sd, Vt


def l_curve(U, sm, b):
    """
    Plot the L-curve and find its "corner".

    Args:
        U (numpy.ndarray): m x min(m, n) matrix.
        sm (numpy.ndarray): min(m, n) x 1 vector.
        b (numpy.ndarray): m x 1 vector containing relaxation data.

    Returns:
        Tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            reg_corner (numpy.ndarray): Regularization parameter at the corner
                                        of the L-curve. Usually of size 1x1.
            rho (numpy.ndarray): Vector of residual norms.
            eta (numpy.ndarray): Vector of solution norms.
            reg_param (numpy.ndarray): Vector of regularization parameters.

    Example:
        reg_corner, rho, eta, reg_param = l_curve(U, s, import_file[:,1])
    """

    # number of regularization parameters to be evaluated
    npoints = 200
    # smallest regularization parameter
    smin_ratio = 16 * np.finfo(float).eps

    # determination of element lengths
    # m, n = U.shape
    p = sm.shape

    # euclidian norm
    beta = np.dot(U.T, b)

    # make beta into p elements
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])

    # solution vector
    xi = np.divide(beta[0:int(p[0])], sm)
    xi[np.isinf(xi)] = 0

    # initialise solvable indices
    rho = np.zeros((npoints, 1))
    eta = np.zeros((npoints, 1))
    reg_param = np.zeros((npoints, 1))

    s2 = sm ** 2

    reg_param[-1] = np.amax([sm[-1], sm[0] * smin_ratio])
    ratio = (sm[0] / reg_param[-1]) ** (1 / (npoints - 1))

    # determine the regularization parameters to check
    for i in np.arange(start=npoints-2, step=-1, stop=-1):
        reg_param[i] = ratio[0] * reg_param[i+1]

    # calculate the norms for all regularization parameters
    for i in np.arange(start=0, step=1, stop=npoints):
        f = s2 / (s2 + reg_param[i] ** 2)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1-f) * beta[:int(p[0])])

    # locate the corner of the L-curve
    reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, sm, b)

    # plot the L-curve
    plot_lc(rho, eta, reg_param, reg_corner, rho_c, eta_c)

    global export_lcurve
    if export_lcurve:
        global data_output

        # determine length of the file format extension
        ffe = len(data_output.rsplit('.', maxsplit=1)[-1]) + 1
        filename = data_output[:-ffe] + '_lcurve' + data_output[-ffe:]

        # save first the optimal parameter, and then the rest.
        np.savetxt(filename, np.r_[[[reg_corner.item(), rho_c, eta_c]],
                                   [[x.item(), y.item(), z.item()] for x, y, z in zip(reg_param, rho, eta)]],
                   header='first line contains lambda_opt, rho_c, eta_c')

    return reg_corner, rho, eta, reg_param


def l_corner(rho, eta, reg_param, U, s, b, order=10):
    """
    Locate the corner of the L-curve in a log-log scale.

    Args:
        rho (numpy.ndarray): Vector of residual norms.
        eta (numpy.ndarray): Vector of solution norms.
        reg_param (numpy.ndarray): Vector of regularization parameters.
        U (numpy.ndarray): m x n matrix containing left singular vectors of A.
        s (numpy.ndarray): Column vector containing singular values of A.
        b (numpy.ndarray): Column vector containing the right-hand side of A*x=b.
        order (int, optional): Order of fitting the 2D spline curve. Defaults to 10.

    Returns:
        Tuple[numpy.ndarray, numpy.float64, numpy.float64]:
            reg_c (numpy.ndarray): Regularization parameter at the corner of the L-curve.
            rho_c (numpy.float64): Vector of residual norms at the corner.
            eta_c (numpy.float64): Vector of solution norms at the corner.

    Example:
        reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, sm, b)
    """

    # transform rho and eta into column vectors (if not already)
    rho = rho.reshape(-1, 1)
    eta = eta.reshape(-1, 1)

    # initialization
    if len(rho) < order:
        raise ValueError('Too little data points for L-curve analysis')

    # m, n = U.shape
    beta = U.T @ b
    xi = beta / s
    xi[np.isinf(xi)] = 0

    # compute the negative curvature of L-curve
    g = lcfun(reg_param, s, beta, xi)

    # locate the corner in the L-curve
    gi = np.argmin(g)
    reg_c = minimize_scalar(lcfun,
                            bounds=(reg_param[min(gi+1, len(g)-1)],
                                    reg_param[max(gi-1, 0)]),
                            args=(s, beta, xi),
                            method='bounded',
                            options={'disp': False}).x
    kappa_max = -lcfun(reg_c, s, beta, xi)

    # if the curvature is negative everywhere,
    # then define the leftmost point of the L-curve as the corner.
    if kappa_max < 0:
        print('No true L-curve found. Negative curvature over full spectrum.')
        lr = len(rho)
        reg_c, rho_c, eta_c = reg_param[lr-1], rho[lr-1], eta[lr-1]
    else:
        f = s ** 2 / (s ** 2 + reg_c ** 2)
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1 - f) * beta[:len(f)])

    return reg_c, rho_c, eta_c


def plot_lc(rho, eta, reg_param, reg_corner, rho_c, eta_c):
    """
    Plot the L-shaped curve of the solution norm.

    The L-curve is a plot of the log of the residual norm ||Ax-b|| versus the
    log of the solution norm ||x|| (assuming ps = 1), where A is a matrix,
    b is a vector, x is the solution vector, and L is a linear operator.

    Args:
        rho (numpy.ndarray): Vector of residual norms.
        eta (numpy.ndarray): Vector of solution or semi-norms.
        reg_param (numpy.ndarray): Vector of regularization parameters.
        reg_corner (numpy.ndarray): Regularization parameter at the corner of the L-curve.
        rho_c (numpy.float64): Vector of residual norms at the corner.
        eta_c (numpy.float64): Vector of solution norms at the corner.

    Returns:
        None

    Example:
        plot_lc(rho, eta, reg_param, reg_corner, rho_c, eta_c)
    """

    # number of identified points
    num_p = 6

    # initialization
    n = len(rho)
    ni = round(n / num_p)

    # make plot
    _, ax = plt.subplots()
    ax.loglog(rho[1:-1], eta[1:-1])
    ax.set_xlim([10 ** np.floor(np.log10(np.min(rho))),
                 10 ** np.ceil(np.log10(np.max(rho)))])
    ax.set_ylim([10 ** np.floor(np.log10(np.min(eta))),
                 10 ** np.ceil(np.log10(np.max(eta)))])

    # plot result
    if max(eta) / min(eta) > 10 or max(rho) / min(rho) > 10:
        # plot loglog
        ax.loglog(rho, eta, rho[ni - 1::ni], eta[ni - 1::ni], 'x')
        for k in range(ni - 1, n, ni):
            ax.text(rho[k], eta[k], str(reg_param[k]))
    else:
        # plot lin
        ax.plot(rho, eta, rho[ni - 1::ni], eta[ni - 1::ni], 'x')
        for k in range(ni - 1, n, ni):
            ax.text(rho[k], eta[k], str(reg_param[k]))

    # plot dotted lines
    plt.hlines(eta_c, 10 ** np.floor(np.log10(np.min(rho))), rho_c,
               colors='red', linestyles='dashed', zorder=0)
    plt.vlines(rho_c, 10 ** np.floor(np.log10(np.min(eta))), eta_c,
               colors='red', linestyles='dashed', zorder=0)

    # set labels and title
    ax.set_xlabel('residual norm $|| A x - b ||_2$')
    ax.set_ylabel('solution norm $|| x ||_2$')
    plt.title('L-curve, corner at ' + str(reg_corner.astype(float))[1:-1])

    plt.show()

    return None


def lcfun(lambda_, s, beta, xi):
    """
    Compute the negative curvature as an auxiliary routine for l_corner.

    Args:
        lambda_ (Union[float, numpy.ndarray]): Regularization parameter,
                                               may be a vector.
        s (numpy.ndarray): Vector of singular values.
        beta (numpy.ndarray): Vector of residuals.
        xi (numpy.ndarray): Vector of solution norms.

    Returns:
        g (numpy.ndarray): The negative curvature. Same size as lambda_.
    """

    # initialization
    phi = np.zeros_like(lambda_)
    dphi = np.zeros_like(lambda_)
    psi = np.zeros_like(lambda_)
    dpsi = np.zeros_like(lambda_)
    eta = np.zeros_like(lambda_)
    rho = np.zeros_like(lambda_)

    # a possible least squares residual
    if len(beta) > len(s):
        LS = True
        rhoLS2 = beta[-1]**2
        beta = beta[:-1]
    else:
        LS = False
        rhoLS2 = 0

    for i, this_lambda_ in enumerate(lambda_):
        if len(xi.shape) > 1:
            f = (s**2)/(s**2 + this_lambda_**2)
        else:
            f = s/(s + this_lambda_)
        cf = 1 - f
        eta[i] = np.linalg.norm(f*xi)
        rho[i] = np.linalg.norm(cf*beta)
        f1 = -2*f*cf/this_lambda_
        f2 = -f1*(3-4*f)/this_lambda_
        phi[i] = np.sum(f*f1*np.abs(xi)**2)
        psi[i] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[i] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[i] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2)

    # take care of a possible least squares residual
    if LS:
        rho = np.sqrt(rho**2 + rhoLS2)

    # now compute the first and second derivatives of eta and rho
    # with respect to lambda
    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)

    # convert to derivatives of log(eta) and log(rho).
    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = ddeta / eta - (dlogeta) ** 2
    ddlogrho = ddrho / rho - (dlogrho) ** 2

    # calculate negative curvature g
    g = -(dlogrho * ddlogeta - ddlogrho * dlogeta) /\
         (dlogrho ** 2 + dlogeta ** 2) ** (1.5)

    return g


def tikhonov(U, s, V, b, lambda_, x_0=None):
    """
    Compute the Tikhonov regularized solution x_lambda given the SVD.

    If the SVD is provided (U, s, V), standard-form regularization is applied:
    min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .

    If an initial estimate x_0 is not specified, x_0 = 0 is used. Note that
    x_0 cannot be used if A is underdetermined and L ~= I.

    If lambda is a vector, x_lambda is a matrix such that
    x_lambda = [ x_lambda(1), x_lambda(2), ... ], The solution norm
    (standard-form case) and the residual norm are returned in eta and rho.

    Args:
        U (numpy.ndarray): Left singular vectors.
        s (numpy.ndarray): Singular values.
        V (numpy.ndarray): Right singular vectors.
        b (numpy.ndarray): Right-hand side vector.
        lambda_ (Union[float, numpy.ndarray]): Regularization parameter(s).
        x_0 (numpy.ndarray, optional): Initial estimate. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            x_lambda (numpy.ndarray): Tikhonov regularized solution(s).
            rho (numpy.ndarray): Residual norm(s).
            eta (numpy.ndarray): Solution norm(s).

    Example:
        x_lambda, rho, eta = tikhonov(U, sd, V, import_file[:,1], reg_corner, x_0)
    """

    # initialization
    if min(lambda_) < 0:
        raise ValueError('Illegal regularization parameter lambda')

    n = V.shape[0]
    p = s.shape[0]
    beta = U[:, :p].T @ b
    zeta = s[:p] * beta
    ll = len(lambda_)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros((ll, 1))
    eta = np.zeros((ll, 1))

    if x_0 is not None:
        omega = V.T @ x_0

    for i in range(ll):
        if x_0 is None:
            x_lambda[:, i] = V[:, :p] * (zeta / (s ** 2 + lambda_[i] ** 2))
            rho[i] = lambda_[i] ** 2 * \
                np.linalg.norm(beta / (s ** 2 + lambda_[i] ** 2))
        else:
            x_lambda[:, i] = V[:, :p] @ ((zeta + lambda_[i]
                                         ** 2 * omega) / (s ** 2 + lambda_[i] ** 2))
            rho[i] = lambda_[i] ** 2 * \
                np.linalg.norm((beta - s * omega) / (s ** 2 + lambda_[i] ** 2))

        eta[i] = np.linalg.norm(x_lambda[:, i])

    if len(np.shape(U)) > 1 and np.shape(U)[1] > p:
        rho = np.sqrt(rho ** 2 + np.linalg.norm(b -
                      U[:, :n] @ np.concatenate((beta, U[:, p:n].T @ b))) ** 2)

    return x_lambda, rho, eta


def import_variable(filename):
    """
    Assign a varriable by importing it from a file.
    Mainly used for debug functionalities.

    Args:
        filename (str): Name of the to be imported file

    Returns:
        output_array (numpy.ndarray): array containing the data.

    Example:
        U = import_variable(U_from_matlab.txt)
    """

    lines = np.loadtxt(filename, dtype='str')
    output_array = []
    for line in lines:
        line = str(line).split(',')
        output_array.append([float(arg) for arg in line])
    return np.array(output_array)


def main(data_input, data_output, plot=True):
    """
    Main entry point for the program.
    Will plot the solution if asked.

    Args:
        data_input (str): Name of file containing input data.
        data_output (str): Name of file where solution will be written to.
        plot (bool, optional): Flag for plotting. Defaults to True.
                               Not in use in current version.

    Returns:
        None

    Example:
        main(data_input, data_output)
    """

    import_file = np.loadtxt(data_input)
    # plt.semilogx(import_file[:,0], import_file[:,1])

    # generate matrix A and logspace (x axis)
    A, sp = discr(1e-6, 1e8, import_file[:, 0])

    # execute singular value decomposition
    U, s, V = csvd(A, False)

    # diagonalize
    if len(s.shape) > 1:
        sd = np.diag(s)
    else:
        sd = s

    # obtain the l-curve and find the l-corner
    reg_corner, *_ = l_curve(U, sd, import_file[:, 1])
    print('Regularization parameter:  %.3e' % reg_corner[0])

    # create empty solution vector
    n = np.size(V, 1)
    x_0 = np.zeros(n)

    # employ Tikhonov analysis
    multiplier = 1e0
    x_lambda, rho, eta = tikhonov(U, sd, V, import_file[:, 1],
                                  reg_corner * multiplier, x_0)

    # print maximum relaxation time
    x_max = np.argmax(x_lambda)
    print('Maximum relaxation time:   %i' % sp[x_max].item())

    # plot the relaxation spectrum
    plt.semilogx(sp, x_lambda)
    plt.show()

    # write the data to the file
    np.savetxt(data_output, [[float(x.item()), float(y.item())]
               for x, y in zip(sp, x_lambda)])

    return None


# ----- MAIN CALL -------------------------------------------------------------
main(data_input, data_output)
