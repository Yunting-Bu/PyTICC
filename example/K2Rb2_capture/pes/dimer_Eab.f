!--------------------------------------------------
!--> The diatomic potential of k2
!--------------------------------------------------
        function v_k2(r)
        implicit none
        integer init_k2
        real*8 v_k2,r,yout
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        common /data_vk2/x,y,y1

        data init_k2/0/
        save init_k2
        if(init_k2==0) then
           call v_k2_init()
           init_k2=1
        endif

        call splint(x,y,y1,n,r,yout)
        v_k2=yout

        return
        end

        subroutine v_k2_init()
        implicit none
        integer i
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        real*8,dimension(n) :: y2,y3,y4
        real*8 dy1,dyn
        common /data_vk2/x,y,y1
        data dy1,dyn/1d30,1d30/

        open(50,file='k2_ab.dat',position='rewind')
        do i=1,n
        read(50,*)x(i),y2(i),y3(i),y4(i),y(i)
        enddo
        close(50)
        call spline(x,y,n,dy1,dyn,y1)
        return
        end subroutine v_k2_init
!--------------------------------------------------

!--------------------------------------------------
!--> The diatomic potential of rb2
!--------------------------------------------------
        function v_rb2(r)
        implicit none
        integer init_rb2
        real*8 v_rb2,r,yout
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        common /data_vrb2/x,y,y1

        data init_rb2/0/
        save init_rb2
        if(init_rb2==0) then
           call v_rb2_init()
           init_rb2=1
        endif

        call splint(x,y,y1,n,r,yout)
        v_rb2=yout

        return
        end

        subroutine v_rb2_init()
        implicit none
        integer i
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        real*8,dimension(n) :: y2,y3,y4
        real*8 dy1,dyn
        common /data_vrb2/x,y,y1
        data dy1,dyn/1d30,1d30/

        open(50,file='rb2_ab.dat',position='rewind')
        do i=1,n
        read(50,*)x(i),y2(i),y3(i),y4(i),y(i)
        enddo
        close(50)
        call spline(x,y,n,dy1,dyn,y1)
        return
        end subroutine v_rb2_init
!--------------------------------------------------

!--------------------------------------------------
!--> The diatomic potential of krb
!--------------------------------------------------
        function v_krb(r)  ! in ang and hartree !
        implicit none
        integer init_krb
        real*8 v_krb,r,yout
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        common /data_vkrb/x,y,y1

        data init_krb/0/
        save init_krb
        if(init_krb==0) then
           call v_krb_init()
           init_krb=1
        endif

        call splint(x,y,y1,n,r,yout)
        v_krb=yout

        return
        end

        subroutine v_krb_init()
        implicit none
        integer i
        integer,parameter :: n=91
        real*8,dimension(n) :: x,y,y1
        real*8,dimension(n) :: y2,y3,y4
        real*8 dy1,dyn
        common /data_vkrb/x,y,y1
        data dy1,dyn/1d30,1d30/

        open(50,file='krb_ab.dat',position='rewind')
        do i=1,n
        read(50,*)x(i),y2(i),y3(i),y4(i),y(i)
        enddo
        close(50)
        call spline(x,y,n,dy1,dyn,y1)
        return
        end subroutine v_krb_init
!--------------------------------------------------
c
cC##################################################################
cC# SPLINE ROUTINES
cC#            Numerical recipes in fortran
cC#            Cambrige University Press
cC#            York, 2nd edition, 1992.
cC##################################################################
c      SUBROUTINE splint(xa,ya,y2a,n,x,y)
c      implicit double precision  (a-h,o-z)
c      DIMENSION xa(n),y2a(n),ya(n)
c      klo=1
c      khi=n
c 5    if (khi-klo.gt.1) then
c        k=(khi+klo)/2
c        if(xa(k).gt.x)then
c          khi=k
c        else
c          klo=k
c        endif
c      goto 5
c      endif
c      h=xa(khi)-xa(klo)
c      if (h.eq.0.0d0) write(6,*) 'bad xa input in splint'
c      a=(xa(khi)-x)/h
c      b=(x-xa(klo))/h
c      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
c     *2)/6.0d0
c      return
c      END
c
cC###################################################################
c      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
c      implicit double precision  (a-h,o-z)
c      DIMENSION x(n),y(n),y2(n)
c      PARAMETER (NMAX=300)
c      DIMENSION u(NMAX)
c      if (yp1.gt..99d30) then
c        y2(1)=0.0d0
c        u(1)=0.0d0
c      else
c        y2(1)=-0.5d0
c        u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
c      endif
c      do 11 i=2,n-1
c        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
c        p=sig*y2(i-1)+2.0d0
c        y2(i)=(sig-1.0d0)/p
c        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+
c     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
c     *u(i-1))/p
c11    continue
c      if (ypn.gt..99d30) then
c        qn=0.0d0
c        un=0.0d0
c      else
c        qn=0.5d0
c        un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
c      endif
c      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
c      do 12 k=n-1,1,-1
c        y2(k)=y2(k)*y2(k+1)+u(k)
c12    continue
c      return
c      END
cC###################################################################
c
c
