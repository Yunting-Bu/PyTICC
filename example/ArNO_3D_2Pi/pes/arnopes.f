!-----------------------------------------------------------------------
!  ArNO PES module. Public version.
!
!  User can use one of four subroutines, given in the table below. Each 
!  subroutine takes a coordinate array (3 bond length or 3 Jacobi 
!  coordinates) as the first argument and returns the potential energies
!  as the second argument (the average and half-difference of A' and A" 
!  or the adiabatic 2x2 ground and excited states).
!
!                    | Bond coords.       | Jacobi coords.
!    ----------------------------------------------------------
!        Va and Vd   | surf_tk_bond       | surf_tk_jac
!      Adiabatic 2x2 | surf_tk_adiab_bond | surf_tk_adiab_jac
!
!    Calling initialization subroutine surf_tk_init is optional, because
!    it is called automatically by the subroutines in the table.
!
!    See headers futher down for more details.
!
!  Authors: Alexander Teplukhin and Brian K. Kendrick
!-----------------------------------------------------------------------

      module arnopes
      implicit none

      ! Parameters
      real*8, parameter   :: pi = 4d0 * atan(1.0d0)
      integer,parameter   :: nsurf = 2  ! Number of surfaces
      real*8, allocatable :: paramd(:)  ! Default parameters
      integer nparam                    ! Number of parameters
      integer :: tbdeg = 5              ! Degree of three-body poly

      ! Masses of N and O to determine CM of NO for Jacobi coords, which
      ! were also used in generating the ab initio data
      real*8, parameter  :: molpro_n  = 14.00307400443d0
      real*8, parameter  :: molpro_o  = 15.99491461957d0
      real*8, parameter  :: molpro_no = 29.997988624d0

      ! Only five subroutines are public
      private
      public :: surf_tk_init
      public :: surf_tk_bond
      public :: surf_tk_jac
      public :: surf_tk_adiab_bond
      public :: surf_tk_adiab_jac

      contains

!-----------------------------------------------------------------------
!  Subroutine surf_tk_bond(rij,v).
!  Calculates Va and Vd in bond coordinates.
!
!  On entering: Array rij contains three bond lengths in bohrs:
!    rij(1) - rNO  - Distance between N  and O
!    rij(2) - rArN - Distance between Ar and N
!    rij(3) - rArO - Distance between Ar and O
!
!  On exit: Array v contains two potential energies in cm-1:
!    v(1) = Va = (A' + A") / 2
!    v(2) = Vd = (A' - A") / 2
!-----------------------------------------------------------------------
      subroutine surf_tk_bond(rij,v)
      implicit none
      real*8 rij(*),v(*)
      real*8 jac(3)
      call bond_to_jac(rij,jac)
      call surf_tk_jac(jac,v)
      end subroutine

!-----------------------------------------------------------------------
!  Subroutine surf_tk_jac(jac,v).
!  Calculates Va and Vd in Jacobi coordinates.
!
!  On entering: Array jac contains three Jacobi coordinates:
!    jac(1) - rNO   - Distance between N  and O in bohrs
!    jac(2) - R     - Distance between Ar and CM of NO in bohrs
!    jac(3) - Theta - Angle between rNO and R in degrees
!
!      Theta =   0 for collinear Ar-NO.
!      Theta = 180 for collinear Ar-ON.
!      Same as in Warehime et al, JCP 142, 024302 (2015)
!      Opposite to Nielson et al, JCP 66, 1396 (1977)
!
!  On exit: Array v, same as in surf_tk_bond.
!-----------------------------------------------------------------------
      subroutine surf_tk_jac(jac,v)
      implicit none
      real*8 jac(*),v(*)
      real*8 jacc(3)
      if(.not.allocated(paramd)) call surf_tk_init
      jacc(1) = max(0.8d0,jac(1))
      jacc(2) = max(4.5d0,jac(2))
      jacc(3) = jac(3)
      call surf_tk_x(paramd,jacc,v)
      end subroutine

!-----------------------------------------------------------------------
!  Subroutine surf_tk_adiab_bond(rij,v).
!  Calculates adiabatic 2x2 version of surfaces in bond coordinates.
! 
!  On entering: Array rij, same as in surf_tk_bond.
!
!  On exit:     Array v, same as in surf_tk_adiab_jac. 
!-----------------------------------------------------------------------
      subroutine surf_tk_adiab_bond(rij,v)
      implicit none
      real*8 rij(*),v(*)
      real*8 jac(3)
      call bond_to_jac(rij,jac)
      call surf_tk_adiab_jac(jac,v)
      end subroutine

!-----------------------------------------------------------------------
!  Subroutine surf_tk_adiab_jac(jac,v).
!  Calculates adiabatic 2x2 version of surfaces in Jacobi coordinates.
!
!  On entering: Array jac, same as in surf_tk_jac.
!
!  On exit:     Array v contains the values of the potential 
!               energy surfaces in cm-1.
! 
!    v(1) = v1   = adiabatic ground state
!    v(2) = v2   = adiabatic excited state
!    v(3) = v11  = diabatic 11 block
!    v(4) = v22  = diabatic 22 block
!    v(5) = v12  = diabatic off-diagonal coupling block
!    v(6) = vdeg = degenerate diagonal diatomic potential
!    v(7) = Aso  = spin orbit coupling constant
!                  (set here and returned for use in calling program)
!
!-----------------------------------------------------------------------
      subroutine surf_tk_adiab_jac(jac,v)
      implicit none

      ! Arguments
      real*8 jac(*),v(*)

      ! Potentials
      real*8 vpa,vpd
      real*8 vp,vm
      real*8 vpno
      real*8 shift

      ! Miscellaneous
      real*8 v1,v2,v11,v22,v12,vdeg,Aso
      real*8 vtk(nsurf)

      ! Call potential
      call surf_tk_jac(jac,vtk)
      vpa = vtk(1)
      vpd = vtk(2)

      ! Subtract NO component, from Va only
      call vNO(paramd(97:99),jac(1),vpno)
      vpa = vpa - vpno

      ! Compute the V+ and V-
      vp = vpa + vpd
      vm = vpa - vpd

      ! Add back NO component
      v11 = vp + vpno
      v22 = vm + vpno

      ! Set spin-orbit splitting constant
      Aso = 123.26d0

      ! Set energy reference to the lower ^2PI_1/2 state
      shift = 0.5d0 * Aso
      v11 = v11 + shift
      v22 = v22 + shift

      ! Set off-diagonal coupling
      v12 = 0.5d0 * Aso

      ! Adiabatic energies, upper state
      v2 = 0.5d0 * ((v11+v22) + sqrt((v11-v22)*(v11-v22) + 4d0*v12*v12))

      ! Adiabatic energies, lower state
      v1 = 0.5d0 * ((v11+v22) - sqrt((v11-v22)*(v11-v22) + 4d0*v12*v12))

      ! Degenerate PI states on diagonal (Vhalf=V3half)
      vdeg = vpno + shift

      ! Pack array
      v(1) = v1
      v(2) = v2
      v(3) = v11
      v(4) = v22
      v(5) = v12
      v(6) = vdeg
      v(7) = Aso

      end subroutine

!-----------------------------------------------------------------------
!  Initialization.
!-----------------------------------------------------------------------
      subroutine surf_tk_init()
      implicit none
      integer i
      character(:),allocatable :: fn
      character(256) pname
      fn = 'parm-3d.in'
      nparam = 99 + get_tb_nparam(tbdeg)
      if(allocated(paramd))deallocate(paramd)
      allocate(paramd(nparam))
      open (1,file=fn)
      do i=1,nparam
         read(1,*)pname,paramd(i)
         if(isnextempty(i))read(1,*)
      enddo
      close(1)
      end subroutine



!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!  The user should not be using the private subprograms below.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!  Computes number of parameters for three-body.
!-----------------------------------------------------------------------
      function get_tb_nparam(d) result(np)
      implicit none
      integer d,np
      integer i,j,k
      np = 3 + 3        ! Three gammas and ref geometry
      do i=0,d
        do j=0,d-i
          do k=0,d-i-j
            np = np + 1
          enddo
        enddo
      enddo
      np = np * nsurf   ! Two surfaces
      end function

!-----------------------------------------------------------------------
!  Checks if the next line is empty.
!-----------------------------------------------------------------------
      logical function isnextempty(i)
      implicit none
      integer i
      if(i == nparam)then
        isnextempty = .false.
      else
        isnextempty = i == 50 .or.
     &                i == 90 .or.
     &                i == 96 .or.
     &                i == 99 .or.
     &                i == 99 + 62
      endif
      end function

!-----------------------------------------------------------------------
!  Potential energy surface routine, takes parameters from array x.
!-----------------------------------------------------------------------
      subroutine surf_tk_x(x,rij,v)
      implicit none
      real*8  x(*),rij(*)
      real*8  rno
      real*8  r
      real*8  theta
      real*8  v(nsurf),tb(nsurf)
      real*8  vnop

      ! Get geometry
      rno   = rij(1)
      r     = rij(2)
      theta = rij(3) * pi / 180d0

      ! Get 2D energy
      call surf_tk_2d(x,r,theta,v)

      ! Add vNO for 3D PES
      call vNO(x(97:99),rno,vnop)
      v(1) = v(1) + vnop

      ! Add TB term
      call vTB(x(   99+1 : 99+62 ),rij,tb(1))
      call vTB(x(99+62+1 : nparam),rij,tb(2))
      v = v + tb
      end subroutine

!-----------------------------------------------------------------------
!  2D potential energy surface as a function of R (bohr) and theta (rad)
!-----------------------------------------------------------------------
      subroutine surf_tk_2d(x,r,theta,v)
      implicit none
      real*8  x(*)
      real*8  r
      real*8  theta
      real*8  v(nsurf)

      ! Constants
      integer, parameter :: np = 3   ! Number of As (or Bs) for one n

      ! Variables
      real*8  scf      ! SCF energy
      real*8  cor      ! COR energy
      real*8  a1,a2,a3 ! A   coefficients
      real*8  b1,b2,b3 ! B   coefficients
      real*8  c6,r0    ! VdW coefficients
      integer nmin     ! Minimum n
      integer nmax     ! Maximum n
      integer n        ! Running n
      integer is       ! State number, 1 or 2
      integer oa,ob,oc ! Offset for As, Bs and Cs

      ! Loop over states
      v = 0d0
      do is=1,nsurf

        ! Setup variables
        if(is == 1)then
          oa   = 0
          ob   = 27
          oc   = 90
          nmin = 0
          nmax = 8
        else
          oa   = 50
          ob   = 50 + 21
          oc   = 90
          nmin = 2
          nmax = 6
        endif

        ! Loop over n and accumulate energy
        do n=nmin,nmax

          ! SCF term
          a1  = x(oa + 1)
          a2  = x(oa + 2)
          a3  = x(oa + 3)
          scf = a1 * exp ( a2 * r + a3 * r**2 )
          oa  = oa + np

          ! COR term
          if(n == 0 .or. n == 2)then
            if(n == 0)then
              c6 = x(oc + 1)
              r0 = x(oc + 4)
            else
              if(is == 1)then
                c6 = x(oc + 2)
                r0 = x(oc + 5)
              else
                c6 = x(oc + 3)
                r0 = x(oc + 6)
              endif
            endif
            b2 = x(ob + 1)
            b1 = c6 / r0**6 * exp( 3d0 + b2 * r0 / 2d0 )
            b3 = - 3d0 / r0**2 - b2 / (2d0 * r0)
            if( r < r0 )then
              cor = - b1 * exp ( b2 * r + b3 * r**2 )
            else
              cor = - c6 / r**6
            endif
            ob = ob + 1
          else
            b1 = x(ob + 1)
            b2 = x(ob + 2)
            b3 = x(ob + 3)
            cor = - b1 * exp ( b2 * r + b3 * r**2 )
            ob = ob + np
          endif

          ! Add up SCF and COR terms
          v(is) = v(is) + (scf + cor) * alp(nmin,n,cos(theta))

        enddo
      enddo
      end subroutine

!-----------------------------------------------------------------------
!  NO potential.
!-----------------------------------------------------------------------
      subroutine vNO(x,r,v)
      implicit none
      real*8 x(*),r,v
      real*8 de,re,be
      de = x(1)
      re = x(2)
      be = x(3)
      v  = de * ( exp( - be * (r - re) ) - 1d0 ) ** 2
      end subroutine

!-----------------------------------------------------------------------
!  Three-body potential.
!-----------------------------------------------------------------------
      subroutine vTB(x,rj,v)
      implicit none
      real*8 x(*),rj(*),v
      real*8 rb(3),ref(3),dr(3),q(3)
      integer d,i,j,k,l

      ! Get geometries and dr
      call jac_to_bond(rj,rb)
      call jac_to_bond(x(1:3),ref)
      dr = rb - ref

      ! Get Q
      q(1) =       (dr(1) + dr(2) + dr(3)) / sqrt(3d0)
      q(2) =               (dr(2) - dr(3)) / sqrt(2d0)
      q(3) = (2d0 * dr(1) - dr(2) - dr(3)) / sqrt(6d0)

      ! Loop over terms
      v = 0d0
      l = 6 ! Skip ref geometry and three gammas
      do d=0,tbdeg
        do i=0,d
          do j=0,d-i
            k = d - i - j
            l = l + 1
            v = v + x(l) * q(1)**i * q(2)**j * q(3)**k
          enddo
        enddo
      enddo

      ! Multiply by tanh with gammas
      v = v * (1d0 - tanh(x(4) * dr(1)))
     &      * (1d0 - tanh(x(5) * dr(2)))
     &      * (1d0 - tanh(x(6) * dr(3)))
      end subroutine

!-----------------------------------------------------------------------
!  Converts Jacobi coordinates to bond lengths.
!    1. rNO                 1. rNO
!    2. R (big R)    -->    2. rArN
!    3. Theta (deg)         3. rArO
!-----------------------------------------------------------------------
      subroutine jac_to_bond(rj,rb)
      implicit none
      real*8 rj(*),rb(*)
      real*8 rno,rl,th
      real*8 rn,ro
      rno   = rj(1)
      rl    = rj(2)
      th    = rj(3) * pi / 180d0
      rn    = rno * molpro_o / molpro_no
      ro    = rno * molpro_n / molpro_no
      rb(1) = rno
      rb(2) = sqrt(rn**2 + rl**2 - 2d0 * rn * rl * cos(th))
      rb(3) = sqrt(ro**2 + rl**2 + 2d0 * ro * rl * cos(th))
      end subroutine

!-----------------------------------------------------------------------
!  Converts bond lengths to bond Jacobi coordinates.
!    1. rNO            1. rNO
!    2. rArN    -->    2. R (big R)
!    3. rArO           3. Theta (deg)
!-----------------------------------------------------------------------
      subroutine bond_to_jac(rb,rj)
      implicit none
      real*8 rj(*),rb(*)
      real*8 rno,rarn,raro
      real*8 rn,a
      rno   = rb(1)
      rarn  = rb(2)
      raro  = rb(3)
      rn    = rno * molpro_o / molpro_no
      a     = acos(clamp((rno**2 + rarn**2 - raro**2) / (2d0*rno*rarn)))
      rj(1) = rno
      rj(2) = sqrt(rn**2 + rarn**2 - 2d0 * rn * rarn * cos(a))
      rj(3) = acos(clamp((rn**2 + rj(2)**2 - rarn**2) / (2d0*rn*rj(2))))
     &        / pi * 180d0
      end subroutine

!-----------------------------------------------------------------------
!  Tiny clamping function. Clamps to [-1; 1].
!-----------------------------------------------------------------------
      real*8 function clamp(t)
      implicit none
      real*8 t
      clamp = t
      if (abs(t) > 1d0) clamp = sign(1d0,t)
      end function

!-----------------------------------------------------------------------
!  Semifactorial function.
!-----------------------------------------------------------------------
      recursive function semifact(n) result(res)
      integer(8) res
      integer, intent(in) :: n
      if (n == 0 .or. n == 1) then
         res = 1
      elseif (n < 0)then
        write(*,*)'Negative argument in semifactorial: ', n
        stop
      else
         res = n * semifact(n-2)
      end if
      end function


!-----------------------------------------------------------------------
!  Associated Legendre polynomial.
!-----------------------------------------------------------------------
      recursive function alp(m,l,x) result(res)
      real*8   res
      integer, intent(in) :: m
      integer, intent(in) :: l
      real*8,  intent(in) :: x
      integer j
      if (l == 0) then
        res = 1
      elseif (m == l) then
        res = (-1)**l * semifact(2 * l - 1) * sqrt(1 - x**2)**l
      elseif (m + 1 == l)then
        res = x * (2 * m + 1) * alp(m,m,x)
      else
        j = l - 1
        res = (2 * j + 1) * x * alp(m,j,x) - (j + m) * alp(m,j-1,x)
        res = res / (j - m + 1) 
      end if
      end function

      end module
