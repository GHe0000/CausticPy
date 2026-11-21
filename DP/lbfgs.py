"""L-BFGS with strong-Wolfe line search."""
import math
import numpy as np
from scipy.optimize import OptimizeResult


def strong_wolfe(evaluate,feasible,x,f,g,p,initial_step,maxls,c1=1e-4,c2=.9):
    derivative0=float(g@p)
    if not derivative0<0:raise ValueError('L-BFGS direction is not a descent direction')
    trials=0;invalid=0
    def trial(alpha):
        nonlocal trials,invalid
        if trials>=maxls:return None
        trials+=1
        xx=x+alpha*p
        if np.array_equal(xx,x):return (xx,None,None)
        if not np.isfinite(xx).all() or (feasible is not None and not feasible(xx)):
            invalid+=1
            return (xx,None,None)
        ff,gg=evaluate(xx)
        return xx,ff,gg
    def acceptable(ff,alpha):
        return ff is not None and ff<=f+c1*alpha*derivative0
    def zoom(lo,hi,flo,fhi,dlo):
        while trials<maxls:
            delta=hi-lo
            alpha=lo+.5*delta
            if fhi is not None:
                denominator=2*(float(fhi-flo)-dlo*delta)
                if denominator!=0 and math.isfinite(denominator):
                    quadratic=lo-dlo*delta**2/denominator
                    a,b=sorted((lo,hi))
                    if a+.1*(b-a)<quadratic<b-.1*(b-a):alpha=quadratic
            if alpha==lo or alpha==hi:return None
            candidate=trial(alpha)
            if candidate is None:return None
            xx,ff,gg=candidate
            if not acceptable(ff,alpha) or ff>=flo:
                hi,fhi=alpha,ff
            else:
                derivative=float(gg@p)
                if abs(derivative)<=-c2*derivative0:
                    return alpha,xx,ff,gg
                if derivative*(hi-lo)>=0:hi,fhi=lo,flo
                lo,flo,dlo=alpha,ff,derivative
        return None
    previous,previous_f,previous_derivative=0.,f,derivative0
    alpha=initial_step
    found=None
    while trials<maxls:
        candidate=trial(alpha)
        if candidate is None:break
        xx,ff,gg=candidate
        if not acceptable(ff,alpha) or (previous>0 and ff>=previous_f):
            found=zoom(previous,alpha,previous_f,ff,previous_derivative);break
        derivative=float(gg@p)
        if abs(derivative)<=-c2*derivative0:
            found=(alpha,xx,ff,gg);break
        if derivative>=0:
            found=zoom(alpha,previous,ff,previous_f,derivative);break
        previous,previous_f,previous_derivative=alpha,ff,derivative
        alpha*=2
    return found,trials,invalid


def minimize_lbfgs(objective,x0,feasible=None,maxiter=1000,gtol=1e-5,ftol=1e-12,
                   maxls=50,maxcor=10,callback=None,initial_step=0.1):
    x=np.asarray(x0,float).copy();history=[]
    evaluations=0;iterations=0;invalid_trials=0;total_trials=0
    best={};evaluation_iteration=0
    if feasible is not None and not feasible(x):raise ValueError('Infeasible L-BFGS initial point')
    def evaluate(xx):
        nonlocal evaluations
        value,gradient=objective(xx);gradient=np.asarray(gradient,float)
        evaluations+=1
        if not math.isfinite(value) or gradient.shape!=xx.shape or not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite objective or gradient at feasible point')
        if not best or value<best['fun']:
            best.update(x=xx.copy(),fun=value,jac=gradient.copy(),iteration=evaluation_iteration,
                        evaluation=evaluations)
        return value,gradient
    f,g=evaluate(x)
    def result(success,status,message):
        return OptimizeResult(x=x,fun=f,jac=g,nit=iterations,nfev=evaluations,success=success,
                              status=status,message=message,invalid_trials=invalid_trials,
                              line_search_trials=total_trials,best_x=best['x'],best_fun=best['fun'],
                              best_jac=best['jac'],best_iteration=best['iteration'])
    if np.max(abs(g))<=gtol:return result(True,0,'Gradient tolerance satisfied')
    for iteration in range(maxiter):
        evaluation_iteration=iteration+1
        q=g.copy();alphas=[]
        for s,y,rho in reversed(history):
            alpha=rho*float(s@q);alphas.append(alpha);q-=alpha*y
        if history:
            s,y,_=history[-1]
            q*=float(s@y)/float(y@y)
        for (s,y,rho),alpha in zip(history,reversed(alphas)):
            beta=rho*float(y@q);q+=s*(alpha-beta)
        direction=-q
        if not float(g@direction)<0:return result(False,2,'Non-descent L-BFGS direction; stopped without fallback')
        initial=min(initial_step,1./np.linalg.norm(g)) if iteration==0 else initial_step
        accepted,trials,invalid=strong_wolfe(evaluate,feasible,x,f,g,direction,initial,maxls)
        total_trials+=trials;invalid_trials+=invalid
        if accepted is None:return result(False,2,'Strong-Wolfe line search did not find a valid step; no zero-step convergence')
        alpha,xnew,fnew,gnew=accepted
        s=xnew-x;y=gnew-g;ys=float(y@s)
        if not np.any(s):return result(False,2,'Floating-point step has zero displacement')
        if ys>np.finfo(float).eps*np.linalg.norm(s)*np.linalg.norm(y):
            history.append((s,y,1./ys))
            if len(history)>maxcor:history.pop(0)
        oldf=f;x,f,g=xnew,fnew,gnew;iterations=iteration+1
        if callback is not None:
            try:callback(x)
            except StopIteration:return result(False,99,'Callback requested stop at accepted iterate')
        if np.max(abs(g))<=gtol:return result(True,0,'Gradient tolerance satisfied')
        if ftol>0 and abs(oldf-f)<=ftol*max(1.,abs(float(oldf)),abs(float(f))):
            return result(True,0,'Relative energy reduction tolerance satisfied after nonzero strong-Wolfe step')
    return result(False,1,'L-BFGS iteration budget exhausted')
