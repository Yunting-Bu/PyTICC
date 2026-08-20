
        module paraspl
        implicit none

        integer,parameter :: nr1=37
        integer,parameter :: nr2=21
        integer,parameter :: nth=5
        real*8 V(nr1,nr2,nth)
        real*8 r1r1(nr1)
        real*8 r2r2(nr2)
        real*8 thth(nth)

        real*8 y2_theta(nr1,nr2,nth)

        end module paraspl


!        program PES_H2_Ar
!        implicit none
!
!        real*8 R,rH2,theta,Vint,t0,t1
!
!        rH2=1.522d0!1.448739d0*0.529177249d0
!        theta=0.d0
!        call cpu_time(t0)
!        do R=3.d0*0.529177249d0,20.d0*0.529177249d0,0.1d0*0.529177249d0
!        do theta=0.d0,180.d0,1.d0
!           call PESH2_Ar(R,rH2,theta,Vint,1)
!           write(111,*)R/0.529177249d0,theta,Vint
!        enddo;enddo
!        call cpu_time(t1)
!        write(*,*)t1-t0
!
!        endprogram


        subroutine PESH2_Ar(R,rH2,theta,Vint,ju)
        implicit none

        integer,save :: init=0
        real*8 R,rH2,theta,Vint
        real*8 r1,r2,th
        real*8,parameter :: bohrtoang=0.529177249d0
        real*8,parameter :: autocm=219474.63137d0
        integer ju

        if(init==0)then
           call readval(ju)
           init=1
        endif

        if(theta.gt.90.d0)then
           th=180.d0-theta
        else
           th=theta
        endif

        r1=R*bohrtoang;r2=rH2*bohrtoang
        call spl3(r1,r2,th,Vint)

        Vint=Vint/autocm

        endsubroutine


        subroutine readval(ju)
        use paraspl
        implicit none

        integer i,j,ju
        real*8 r1

        thth(1)=0.d0;thth(2)=25.873724975d0;thth(3)=47.37584165d0
        thth(4)=68.708225955d0;thth(5)=90.d0

        do i=1,nr2,1
           r2r2(i)=0.3d0+0.1d0*dble(i-1)
        enddo

        if(ju==0)then
           open(unit=111,file='nore.dat',status='old')
        elseif(ju==1)then
           open(unit=111,file='re.dat',status='old')
        endif

        do i=1,4,1
           read(111,*)
        enddo

        do i=1,nr2,1
           do j=1,nr1,1
              if(i==1)then
                 read(111,*)r1r1(j),V(j,i,1:5)
              else
                 read(111,*)r1,V(j,i,1:5)
              endif
!              write(*,*)i,j,V(j,i,1)
           enddo
           if(i.lt.nr2)then
              read(111,*)
           endif
        enddo

        close(111)

        do i=1,nr1,1
           do j=1,nr2,1
              call spline(thth,V(i,j,:),nth,0.d0,0.d0,y2_theta(i,j,:))
           enddo
        enddo

        endsubroutine


        subroutine spl3(r1,r2,th,Vint)
        use paraspl
        implicit none

        real*8,intent(in) :: r1,th,r2
        real*8,intent(out) :: Vint

        integer i,j,k
        real*8 ss(nr2),sss(nr1)
        real*8 y2tmp(nr2),y2tmp2(nr1) !for r2 and r1
        real*8 val

        do i=1,nr1
           do j=1,nr2
              call splint(thth,V(i,j,:),y2_theta(i,j,:),nth,th,val)
              ss(j)=val
           enddo
           call spline(r2r2,ss,nr2,1.d30,1.d30,y2tmp)
           call splint(r2r2,ss,y2tmp,nr2,r2,val)
           sss(i)=val
        enddo

        call spline(r1r1,sss,nr1,1.d30,1.d30,y2tmp2)
        call splint(r1r1,sss,y2tmp2,nr1,r1,Vint)

        return
        end subroutine spl3


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
