        module paraspl
        implicit none

        integer,parameter :: nrA=17
        integer,parameter :: ntot=11
        integer init
        real*8 ele(ntot,nrA)
        real*8 rArA(nrA)!grid

        end module paraspl

        subroutine spl1(r1,elee)
        use paraspl
        implicit none

        real*8 r1
        real*8 elee(ntot)

        integer iop,i,j,k

        integer,parameter :: m=100
        real*8 y3
        real*8 xt(m)
        real*8 y(m),y2(m)
        real*8,parameter :: dy1=1.0d30
        real*8,parameter :: dyn=1.0d30

        if(init==0)then
           init=1
           open(unit=30,file='ele_pro.dat',status='old')
           do i=1,nrA,1
              read(30,*)rArA(i),ele(1:11,i)
           enddo
        endif

        do iop=1,ntot,1
        do i=1,nrA,1
        xt(i)=rArA(i)
        y(i)=ele(iop,i)
        enddo
        call spline(xt,y,nrA,dy1,dyn,y2)
        call splint(xt,y,y2,nrA,r1,y3)
        elee(iop)=y3
        enddo

        endsubroutine


        !##################################################################
        !# SPLINE ROUTINES
        !#            Numerical recipes in fortran
        !#            Cambrige University Press
        !#            York, 2nd edition, 1992.
        !##################################################################

        SUBROUTINE spline(x,y,n,yp1,ypn,y2)
        implicit double precision  (a-h,o-z)
        DIMENSION x(n),y(n),y2(n)
        PARAMETER (NMAX=100)
        DIMENSION u(NMAX)
        if (yp1.gt..99d30) then
          y2(1)=0.0d0
          u(1)=0.0d0
        else
          y2(1)=-0.5d0
          u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
        endif
        do 11 i=2,n-1
          sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
          p=sig*y2(i-1)+2.0d0
          y2(i)=(sig-1.0d0)/p
          u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)
     &         -x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
11      continue
        if (ypn.gt..99d30) then
          qn=0.0d0
          un=0.0d0
        else
          qn=0.5d0
          un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
        endif
        y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
        do 12 k=n-1,1,-1
          y2(k)=y2(k)*y2(k+1)+u(k)
12      continue
        return
        END

        SUBROUTINE splint(xa,ya,y2a,n,x,y)
        implicit double precision  (a-h,o-z)
        DIMENSION xa(n),y2a(n),ya(n)
        klo=1
        khi=n
1       if (khi-klo.gt.1) then
          k=(khi+klo)/2
          if(xa(k).gt.x)then
            khi=k
          else
            klo=k
          endif
        goto 1
        endif
        h=xa(khi)-xa(klo)
        if (h.eq.0.0d0) write(6,*) 'bad xa input in splint'
        a=(xa(khi)-x)/h
        b=(x-xa(klo))/h
        y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))
     &    *(h**2)/6.0d0
        return
        END

