def eq(y, t, N, beta, gamma, delta, alpha, omega, zeta):
  S, E, I, R, D = y
  dSdt = -beta*S*I/N + (t/125)*(delta*E) - (t/125)*(gamma*I) - (t/125)*(alpha*I) - alpha * I
  dEdt = beta*S*I/N - delta*E
  dIdt = (t/omega)*(delta*E) - (t/omega)*(gamma*I) - (t/omega)*(alpha*I) # t - change
  dRdt = gamma*I*zeta # - change
  dDdt = alpha*I # +-: shift v */:scale amplitude
  return dSdt, dEdt, dIdt, dRdt, dDdt