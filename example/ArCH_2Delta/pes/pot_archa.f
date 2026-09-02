c-----------------------------------------------------------------------
c  CH(A 2Delta)-Ar potential energy surfaces
c
c  Reference: G. Kerenskaya, A. L. Kaledin, M. C. Heaven,
c             J. Chem. Phys. 115, 2123 (2001); doi:10.1063/1.1382647
c
c  Analytic representation (Eqs. 1a, 1b, 16 and Table IV of the paper):
c
c    V_a(R,g) = sum_{l=0}^{6} V_l0(R) * Pbar_l0(cos g)
c    V_d(R,g) = sum_{l=4}^{6} V_l4(R) * Pbar_l4(cos g)
c
c    V_lm(R) = D_lm [ exp(-2 a_lm (R-R_lm)) - 2 exp(-a_lm (R-R_lm)) ]
c
c  with Pbar the Racah-normalized associated Legendre functions and
c  m = 2*Lambda = 4 for the difference potential.
c
c  Lambda = 2  =>  (-1)^Lambda = +1, so
c      V_A'  = V_a + V_d
c      V_A'' = V_a - V_d
c
c  Input:  r in Angstrom, gamma in radians (gamma=0 -> linear CH-Ar,
c  gamma=pi -> linear HC-Ar).  Output: energies in cm^-1.
c
c  isurf = 0 : original ab initio surface (Table IV parameters)
c  isurf = 1 : empirical surface 1 (D scale: V00*1.62, V10*0.2, V20*1.568)
c  isurf = 2 : empirical surface 2 (D scale: V00*1.587, V10*0.877,
c              V20*1.221; alpha of V10 scaled by 0.698)
c
c  dre (Angstrom): uniform shift applied to all R_e parameters.
c  The paper recommends dre = -0.25 to match rotational constants.
c-----------------------------------------------------------------------
      subroutine chad_pes(r, gamma, isurf, dre, va, vd)
      implicit double precision (a-h,o-z)
      parameter (n=10)
      dimension d(n), re(n), al(n), s(n), sa(n), sa2(n)
      data d /
     *    82.266d0,   9.048d0,  31.035d0,   4.845d0,   3.057d0,
     *     0.931d0,   0.374d0,3535.5d0, -83.115d0,  40.911d0/
      data re/
     *     3.982d0,   4.076d0,   3.499d0,   4.006d0,   3.805d0,
     *     3.956d0,   3.862d0,   1.255d0,   1.306d0,   1.526d0/
      data al/
     *     1.524d0,   1.771d0,   2.115d0,   1.763d0,   1.914d0,
     *     1.752d0,   1.867d0,   2.703d0,   2.404d0,   2.678d0/
c     D scale factors: surface 1 and surface 2 (Table IV)
      data sa/
     *     1.62d0,   0.2d0,    1.568d0,  1.0d0,    1.0d0,
     *     1.0d0,    1.0d0,    1.0d0,    1.0d0,    1.0d0/
      data sa2/
     *     1.587d0,  0.877d0,  1.221d0,  1.0d0,    1.0d0,
     *     1.0d0,    1.0d0,    1.0d0,    1.0d0,    1.0d0/
      save d, re, al, sa, sa2

      va = 0.d0
      vd = 0.d0
      x = cos(gamma)
c     Legendre polynomials P_l(x) for l = 0..6
      p0 = 1.d0
      p1 = x
      p2 = 0.5d0*(3.d0*x*x - 1.d0)
      p3 = 0.5d0*(5.d0*x**3 - 3.d0*x)
      p4 = 0.125d0*(35.d0*x**4 - 30.d0*x*x + 3.d0)
      p5 = 0.125d0*(63.d0*x**5 - 70.d0*x**3 + 15.d0*x)
      p6 = 0.0625d0*(231.d0*x**6 - 315.d0*x**4 + 105.d0*x*x - 5.d0)
c     associated Legendre P_l^4(x) for l = 4, 5, 6
      sm2 = (1.d0 - x*x)**2
      pa4 = 105.d0*sm2
      pa5 = 9.d0*x*pa4
      pa6 = (13.d0*x*pa5 - 8.d0*pa4)/3.d0
c     Racah normalization: Pbar_lm = sqrt((2l+1)(l-m)!/(2(l+m)!)) P_l^m
      pa4 = pa4*0.0105651d0
      pa5 = pa5*0.0038933d0
      pa6 = pa6*0.0018928d0

      do i = 1, n
        s(i) = 1.d0
      enddo
      if (isurf .eq. 1) then
        do i = 1, n
          s(i) = sa(i)
        enddo
      elseif (isurf .eq. 2) then
        do i = 1, n
          s(i) = sa2(i)
        enddo
      endif

      do i = 1, n
        rr = re(i) + dre
        a = al(i)
        if (isurf .eq. 2 .and. i .eq. 2) a = a*0.698d0
        xarg = exp(-a*(r - rr))
        v = s(i)*d(i)*(xarg*xarg - 2.d0*xarg)
        if (i .le. 7) then
          if (i .eq. 1) va = va + v*p0
          if (i .eq. 2) va = va + v*p1
          if (i .eq. 3) va = va + v*p2
          if (i .eq. 4) va = va + v*p3
          if (i .eq. 5) va = va + v*p4
          if (i .eq. 6) va = va + v*p5
          if (i .eq. 7) va = va + v*p6
        else
          if (i .eq. 8)  vd = vd + v*pa4
          if (i .eq. 9)  vd = vd + v*pa5
          if (i .eq. 10) vd = vd + v*pa6
        endif
      enddo
      return
      end