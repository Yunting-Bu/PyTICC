***************************************************************************
        subroutine pesh3f(r1,r2,rr,th1,th2,phi,vint,vtot)
***************************************************************************
c       Subroutine to generate values of the 6D PES for complex 
c       formed between H2 and HF, as determined by:
c       D. Yang, J. Huang, J. Zuo, X. Hu, D. Xie
c       in the paper titled
c      "A new full-dimensional potential energy surface and quantum
c       dynamics of inelastic collision process for H2-HF".
c
c       Input variables:
c         r1, r2 - Two bond lengths for the H2 and HF monomers.
c         rr     - The distance from the centre of mass of H2
C                  to that of HF.
c         th1,th2- Jacobi angular coordinates, which are enclosed
c                  angles between the vectors R and r1(r2).
c         phi    - Jacobi angular coorinate of the dihedral angle.
c
c       Output:
c         vint  -  The calculated interaction energy '\Delta{V}'
c         vtot  -  The calculated total potential energy for the complex.
c
c       Notice the units for distance, angle, and energy are
c       respectively bohr, degree, and wavenumber.
c--------------------------------------------------------------------------
        implicit none
        double precision :: r1,r2,rr,th1,th2,phi,q(3,4),Jcb(6)
        double precision :: vtot,vint,v1,v2,vnn,vlr
        double precision :: D_R
        real*8,parameter :: bohr=0.529177249d0
        real*8,parameter :: tocm=2.1947463137d5
        integer init
        data init/0/
        save init
        if(init==0) then
           call nnpes_init
           init=1
        endif
        Jcb(1)=rr;Jcb(2)=r1;Jcb(3)=r2
        Jcb(4)=th1;Jcb(5)=th2;Jcb(6)=phi
! q(3,4) in HHHF order and bohr
        call bf_to_xyz(r1,r2,rr,th1,th2,phi,q)
        call h2hfNN(q,vnn)
        call long_range_interaction(Jcb,vlr)    
        D_R=1.d0/(1.d0+dexp(-2.d0*(rr-14.d0)))
        Vint=(1.d0-D_R)*vnn+D_R*vlr
        vint=vint/tocm
!        vint=vlr

        call vh2(r1,v1)
        call vhf(r2,v2)
        vtot=vint+v1+v2 ! all in hartree

        return
        end subroutine pesh3f
!#########################################################################
!-->  program to get potential energy for a given geometry after NN
!fitting
!-->  global variables are declared in this module
        module nnparam
        implicit none
        integer ninput,noutput,nhid,nlayer,ifunc,nwe,nodemax
        integer nscale
        integer, allocatable::nodes(:)
        real*8, allocatable::weighta(:,:,:),biasa(:,:)
        real*8, allocatable::pdela(:),pavga(:)
        real*8, allocatable::weightb(:,:,:),biasb(:,:)
        real*8, allocatable::pdelb(:),pavgb(:)
        real*8, allocatable::weightc(:,:,:),biasc(:,:)
        real*8, allocatable::pdelc(:),pavgc(:)
!        double precision,parameter:: vpescut=5.0d0
        end module nnparam
!##########################################################################



!**************************************************************************
! ct in HHHF order and bohr; vpes in cm-1.
        subroutine h2hfNN(ct,vpes) 
        use nnparam
        implicit none
        integer i
        real*8 rb(6),xbond(6),basis(0:17)
        real*8 txinput(17)
        real*8 ct(3,4),xvec(3,6),xct(3,4)
        real*8 va,vb,vc,vpes,dv
        double precision,parameter :: bohr=0.529177249d0
        double precision,parameter :: alpha=2.5d0 !Ang
        double precision,parameter :: Vcut=2.d4 ! cm-1
 
        basis=0.d0
        xct=ct*bohr ! bohr to Ang

        xvec(:,1)=xct(:,2)-xct(:,1) !  H2->H1
        xvec(:,2)=xct(:,3)-xct(:,1) !  H3->H1
        xvec(:,3)=xct(:,4)-xct(:,1) !  F ->H1
        xvec(:,4)=xct(:,3)-xct(:,2) !  H3->H2
        xvec(:,5)=xct(:,4)-xct(:,2) !  F ->H2
        xvec(:,6)=xct(:,4)-xct(:,3) !  F ->H3 

        rb(1)=dsqrt(dot_product(xvec(:,1),xvec(:,1)))
        rb(2)=dsqrt(dot_product(xvec(:,2),xvec(:,2)))
        rb(3)=dsqrt(dot_product(xvec(:,3),xvec(:,3)))
        rb(4)=dsqrt(dot_product(xvec(:,4),xvec(:,4)))
        rb(5)=dsqrt(dot_product(xvec(:,5),xvec(:,5)))
        rb(6)=dsqrt(dot_product(xvec(:,6),xvec(:,6)))
 
        xbond(:)=dexp(-rb(:)/alpha)
 
        call bemsav(xbond,basis)
 
        do i=1,17
           txinput(i)=basis(i)
        enddo
 
        call getpota(txinput,va)
        call getpotb(txinput,vb)
        call getpotc(txinput,vc)
        vpes=(va+vb+vc)/3.0d0

        dv=max(va,vb,vc)-min(va,vb,vc)
        if (dv .gt. 50.d0) then 
           vpes=Vcut
        endif
        vpes=min(vpes,Vcut)
 
        return
        end subroutine h2hfNN
!**************************************************************************


!**************************************************************************
!-->  read NN weights and biases from matlab output
!-->  weights saved in 'weights.txt'
!-->  biases saved in 'biases.txt'
!-->  one has to call this subroutine one and only one before calling
!the getpot() subroutine
        subroutine nnpes_init
        use nnparam
        implicit none
        integer ihid,iwe,inode1,inode2,ilay1,ilay2
        integer i,wfilea,bfilea,wfileb,bfileb,wfilec,bfilec
        
        wfilea=551
        bfilea=661 
        wfileb=552
        bfileb=662
        wfilec=553
        bfilec=663 

        open(wfilea,file='weightsa.txt',status='old')
        open(bfilea,file='biasesa.txt',status='old')

        rewind(wfilea)
        rewind(bfilea)

        read(wfilea,*)ninput,nhid,noutput
        nscale=ninput+noutput
        nlayer=nhid+2 
        allocate(nodes(nlayer),pdela(nscale),pavga(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(wfilea,*)(nodes(ihid),ihid=2,nhid+1)
        nodemax=0
        do i=1,nlayer
           nodemax=max(nodemax,nodes(i))
        enddo
        allocate(weighta(nodemax,nodemax,2:nlayer),
     %  biasa(nodemax,2:nlayer))
        read(wfilea,*)ifunc,nwe
        read(wfilea,*)(pdela(i),i=1,nscale)
        read(wfilea,*)(pavga(i),i=1,nscale)
        iwe=0
        do ilay1=2,nlayer
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
               do inode2=1,nodes(ilay2) 
                   read(wfilea,*)weighta(inode2,inode1,ilay1)
                   iwe=iwe+1
               enddo
               read(bfilea,*)biasa(inode1,ilay1)
               iwe=iwe+1
           enddo
        enddo
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif
        close(wfilea)
        close(bfilea)
        

        open(wfileb,file='weightsb.txt',status='old')
        open(bfileb,file='biasesb.txt',status='old')

        rewind(wfileb)
        rewind(bfileb)

        read(wfileb,*)ninput,nhid,noutput
        nscale=ninput+noutput
        nlayer=nhid+2
        allocate(pdelb(nscale),pavgb(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(wfileb,*)(nodes(ihid),ihid=2,nhid+1)
        nodemax=0
        do i=1,nlayer
           nodemax=max(nodemax,nodes(i))
        enddo
        allocate(weightb(nodemax,nodemax,2:nlayer),
     %  biasb(nodemax,2:nlayer))
        read(wfileb,*)ifunc,nwe
        read(wfileb,*)(pdelb(i),i=1,nscale)
        read(wfileb,*)(pavgb(i),i=1,nscale)

        iwe=0
        do ilay1=2,nlayer
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
               do inode2=1,nodes(ilay2)
                   read(wfileb,*)weightb(inode2,inode1,ilay1)
                   iwe=iwe+1
               enddo
               read(bfileb,*)biasb(inode1,ilay1)
               iwe=iwe+1
           enddo
        enddo
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif
        close(wfileb)
        close(bfileb)


        open(wfilec,file='weightsc.txt',status='old')
        open(bfilec,file='biasesc.txt',status='old')

        rewind(wfilec)
        rewind(bfilec)

        read(wfilec,*)ninput,nhid,noutput
        nscale=ninput+noutput
        nlayer=nhid+2
        allocate(pdelc(nscale),pavgc(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(wfilec,*)(nodes(ihid),ihid=2,nhid+1)
        nodemax=0
        do i=1,nlayer
           nodemax=max(nodemax,nodes(i))
        enddo
        allocate(weightc(nodemax,nodemax,2:nlayer),
     %  biasc(nodemax,2:nlayer))
        read(wfilec,*)ifunc,nwe
        read(wfilec,*)(pdelc(i),i=1,nscale)
        read(wfilec,*)(pavgc(i),i=1,nscale)

        iwe=0
        do ilay1=2,nlayer
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
               do inode2=1,nodes(ilay2)
                   read(wfilec,*)weightc(inode2,inode1,ilay1)
                   iwe=iwe+1
               enddo
               read(bfilec,*)biasc(inode1,ilay1)
               iwe=iwe+1
           enddo
        enddo
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif
        close(wfilec)
        close(bfilec)

        return

        end subroutine nnpes_init
!**************************************************************************


!**************************************************************************
        subroutine getpota(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun

!-->....set up the normalized input layer
        do i=1,ninput
           y(i,1)=(x(i)-pavga(i))/pdela(i)
        enddo

!-->....evaluate the hidden layer
        do ilay1=2,nlayer-1
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
              y(inode1,ilay1)=biasa(inode1,ilay1)
              do inode2=1,nodes(ilay2)
                 y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                           *weighta(inode2,inode1,ilay1)
              enddo
              y(inode1,ilay1)=tranfun(y(inode1,ilay1),ifunc)
           enddo
        enddo

!-->....now evaluate the output
        ilay1=nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
           y(inode1,ilay1)=biasa(inode1,ilay1)
           do inode2=1,nodes(ilay2)
              y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                        *weighta(inode2,inode1,ilay1)
           enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdela(nscale)+pavga(nscale)
        return
        end subroutine getpota
!**************************************************************************


!**************************************************************************
        subroutine getpotb(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun

!-->....set up the normalized input layer
        do i=1,ninput
           y(i,1)=(x(i)-pavgb(i))/pdelb(i)
        enddo

!-->....evaluate the hidden layer
        do ilay1=2,nlayer-1
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
              y(inode1,ilay1)=biasb(inode1,ilay1)
              do inode2=1,nodes(ilay2)
                 y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                           *weightb(inode2,inode1,ilay1)
              enddo
              y(inode1,ilay1)=tranfun(y(inode1,ilay1),ifunc)
           enddo
        enddo

!-->....now evaluate the output
        ilay1=nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
           y(inode1,ilay1)=biasb(inode1,ilay1)
           do inode2=1,nodes(ilay2)
              y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                        *weightb(inode2,inode1,ilay1)
           enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdelb(nscale)+pavgb(nscale)
        return
        end subroutine getpotb
!**************************************************************************


!**************************************************************************
        subroutine getpotc(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun

!-->....set up the normalized input layer
        do i=1,ninput
           y(i,1)=(x(i)-pavgc(i))/pdelc(i)
        enddo

!-->....evaluate the hidden layer
        do ilay1=2,nlayer-1
           ilay2=ilay1-1
           do inode1=1,nodes(ilay1)
              y(inode1,ilay1)=biasc(inode1,ilay1)
              do inode2=1,nodes(ilay2)
                 y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                           *weightc(inode2,inode1,ilay1)
              enddo
              y(inode1,ilay1)=tranfun(y(inode1,ilay1),ifunc)
           enddo
        enddo

!-->....now evaluate the output
        ilay1=nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
           y(inode1,ilay1)=biasc(inode1,ilay1)
           do inode2=1,nodes(ilay2)
              y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     &                        *weightc(inode2,inode1,ilay1)
           enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdelc(nscale)+pavgc(nscale)
        return
        end subroutine getpotc
!**************************************************************************


!**************************************************************************
        function tranfun(x,ifunc)
        implicit none
        integer ifunc
        real*8 tranfun,x
c    ifunc=1, transfer function is hyperbolic tangent function, 'tansig'
c    ifunc=2, transfer function is log sigmoid function, 'logsig'
c    ifunc=3, transfer function is pure linear function, 'purelin'. It 
c             is imposed to the output layer by default
        if (ifunc.eq.1) then
           tranfun=dtanh(x)
        else if (ifunc.eq.2) then
           tranfun=1d0/(1d0+exp(-x))
        else if (ifunc.eq.3) then
           tranfun=x
        endif
        return
        end
!**************************************************************************


!********************************************************
        subroutine bemsav(x,p)
        implicit none
        double precision,intent(in) :: x(1:6)
        double precision,intent(out) :: p(0:17)
        double precision :: m(0:10)

        call evmono(x,m)
        call evpoly(m,p)

        return
        end subroutine bemsav
!********************************************************
!********************************************************
        subroutine evmono(x,m)
        implicit none
        double precision,intent(in) :: x(1:6)
        double precision,intent(out) :: m(0:10)

        m(0)=1.d0
        m(1)=x(6)
        m(2)=x(5)
        m(3)=x(3)
        m(4)=x(4)
        m(5)=x(2)
        m(6)=x(1)
        m(7)=m(2)*m(3)
        m(8)=m(3)*m(4)
        m(9)=m(2)*m(5)
        m(10)=m(4)*m(5)

        return
        end subroutine evmono
!********************************************************

!********************************************************
        subroutine evpoly(m,p)
        implicit none
        double precision,intent(in) :: m(0:10)
        double precision,intent(out) :: p(0:17)

        p(0)=m(0)
        p(1)=m(1)
        p(2)=m(2)+m(3)
        p(3)=m(4)+m(5)
        p(4)=m(6)
        p(5)=p(1)*p(2)
        p(6)=m(7)
        p(7)=p(1)*p(3)
        p(8)=m(8)+m(9)
        p(9)=m(10)
        p(10)=p(2)*p(3)-p(8)
        p(11)=p(1)*p(4)
        p(12)=p(4)*p(2)
        p(13)=p(4)*p(3)
        p(14)=p(1)*p(1)
        p(15)=p(2)*p(2)-p(6)-p(6)
        p(16)=p(3)*p(3)-p(9)-p(9)
        p(17)=p(4)*p(4)

        return
        end subroutine evpoly
!********************************************************
        subroutine vh2(r,v)
        implicit none
        double precision :: r,p,v
        double precision, parameter :: de=4.747d0,re=0.7414d0 ! eV and Angstrom
        double precision, parameter :: a1=3.961d0,a2=4.064d0,a3=3.574d0
        double precision, parameter :: bohr=0.529177249d0
        double precision, parameter :: toev=27.2114d0

        p=r*bohr-re
        v=de-de*(1+a1*p+a2*p**2+a3*p**3)*dexp(-a1*p)
        v=v/toev        ! eV --> hartree

        return
        end subroutine vh2

        subroutine vhf(r,v)
        implicit none
        integer :: i
        double precision :: vtmp,v,r,p
        double precision :: a(6)
        double precision, parameter :: de=49361.6d0 ! cm^-1
        double precision, parameter :: re=1.7325d0 ! bohr 
        double precision, parameter :: bohr=0.529177249d0
        double precision, parameter :: tocm=2.1947463137d5

        a=(/2.23729d0 ,1.12367d0 , 5.68736d-1,
     $      9.18139d-3,8.07740d-3,-3.51111d-2/)

        p=r-re
        vtmp=0.d0
        do i=1,6
           vtmp=vtmp+a(i)*p**i
        enddo

        v=de-de*(1+vtmp)*dexp(-a(1)*p) ! cm^-1
        v=v/tocm

        return
        end subroutine vhf

!--------------------------------------------------------------------
        function funca(l1,l2,l,th1,th2,phi)
        implicit real*8(a-h,o-z)
        COMMON/LNFJ/ SLNI(164),SLNF(164),SLNJ(164)
        real*8,external::s3j,ylm
        real*8,parameter::pi=3.141592653d0
        data init/0/
        save init
        if(init==0) then
        CALL SLNN
        init=1
        endif

        m=min(l1,l2)
        
        nm=2*m+1
        s=0d0
        do i=1,nm
           j=i-m-1

          if(l1==0.and.l2==0.and.l==0.and.m==0) then
             x=1d0
          else
             xj1=dble(l1);xj2=dble(l2);xj3=dble(l)
             xm1=dble(j);xm2=dble(-j)
             x=s3j(xj1,xj2,xj3,xm1,xm2,0.d0)
          endif
          y1=ylm(l1,j,th1)
          y2=ylm(l2,j,th2)
          s1=x*y1*y2*dcos(dble(j)*phi)*(-1.d0)**j

          s=s+s1
        enddo
        s=s*4.d0*pi/dsqrt((2.d0*l1+1.d0)*
     1           (2.d0*l2+1.d0))

        funca=s
        return
        end function

        function ylm(l,m,th)
        implicit none
        real*8 th,x,pi,phi,ylm
        real*8,external::spgndr
        integer l,m
        pi=dacos(-1.d0)
        x=dcos(th)
        ylm=dsqrt(1.d0/2.d0/pi)*spgndr(l,m,x)
        return
        end function

!###################################################################
!SUBROUTINE FOR CALCULATING THE FUNCTIONAL VALUES OF 3-J SYMBLE.
!FROM:
!       COMMON ALGORITHMS AND PROGRAMS IN QUANTUM PHYSICS
!AUTHORS:
!       JING XIAOGONG, ZHAO YONGFANG AND HAO FENGYOU.
!       HARBIN INSTITUDE OF TECHNOLOGY PRESS, 2009.12
!###################################################################
        DOUBLE PRECISION FUNCTION S3J(XJ1,
     *          XJ2,XJ3,XM1,XM2,XM3)
        IMPLICIT DOUBLE PRECISION(A-H,O-Z)
        COMMON/LNFJ/ SLNI(164),SLNF(164),SLNJ(164)
        IF(DABS(XM1).GT.XJ1.OR.DABS(XM2).
     *  GT.XJ2.OR.DABS(XM3).GT.XJ3) GOTO10
        M=IFX(XM1+XM2+XM3)
        JM=IFX(XJ1-XJ2-XM3)
        J12=IFX(XJ1+XJ2-XJ3)
        J13=IFX(XJ1-XJ2+XJ3)
        J23=IFX(-XJ1+XJ2+XJ3)
        J=IFX(XJ1+XJ2+XJ3)
        JM11=IFX(XJ1+XM1)
        JM12=IFX(XJ1-XM1)
        JM21=IFX(XJ2+XM2)
        JM22=IFX(XJ2-XM2)
        JM31=IFX(XJ3+XM3)
        JM32=IFX(XJ3-XM3)
        MJ32=IFX(XJ3-XJ2+XM1)
        MJ31=IFX(XJ3-XJ1-XM2)
        IF(M.NE.0.OR.J12.LT.0.OR.J13.LT.0.
     *          OR.J23.LT.0) GOTO 10
        A=SLNF(J12+1)+SLNF(J13+1)+
     *  SLNF(J23+1)+SLNF(JM11+1)+SLNF
     *  (JM12+1)+SLNF(JM21+1)+SLNF(JM22+1)+
     *  SLNF(JM31+1)+SLNF(JM32+1)-SLNF(J+2)     
        A=DEXP(A)
        A=DSQRT(A)
        B=0.0D0
        DO 1 K=0,10000
        IF(J12-K.LT.0.OR.JM12-K.LT.0.OR.
     *  JM21-K.LT.0.OR.MJ32+K.LT.0.OR.
     *  MJ31+K.LT.0) GOTO 1
        B0=-SLNF(K+1)-SLNF(J12-K+1)-
     *  SLNF(JM12-K+1)-SLNF(JM21-K+1)
     *  -SLNF(MJ32+K+1)-SLNF(MJ31+K+1)
!        IF(DABS(B0).LT. 0.1E-35) GOTO 1
        B0=DEXP(B0)
        IF(K.EQ.0) CK=1.D0
        IF(K.NE.0) CK=(-1.D0)**K
        B=B+CK*B0
1       CONTINUE
        IF(JM.EQ.0) CC=1.D0
        IF(JM.NE.0) CC=(-1.D0)**JM
        S3J=CC*A*B
        GOTO 20
10      S3J=0.0D0
20      CONTINUE
!        WRITE(50,2) XJ1,XJ2,XJ3,XM1,XM2,XM3,S3J
!2       FORMAT(2X,6F7.2,/,5X,'3J=',E15.7)       
        RETURN
        END     
 
        SUBROUTINE SLNN
        IMPLICIT DOUBLE PRECISION(A-H,O-Z)
        COMMON/LNFJ/ SLNI(164),SLNF(164),SLNJ(164)
        DO N=2,164
        SLNI(N)=DLOG(DBLE(FLOAT(N-1)))
        ENDDO
        SLNI(1)=0.0D0   
        FLN=1.0D0
        DO N=2,164
        FLN=FLN*DBLE(FLOAT(N-1))
        SLNF(N)=DLOG(FLN)
        ENDDO
        SLNF(1)=0.0D0
        FLN=1.0D0
        DO N=1,163,2
        FLN=FLN*DBLE(FLOAT(N))
        SLNJ(N+1)=DLOG(FLN)
        ENDDO
        FLN=1.0D0
        DO N=2,162,2
        FLN=FLN*DBLE(FLOAT(N))
        SLNJ(N+1)=DLOG(FLN)
        ENDDO
        SLNJ(1)=0.0D0
        RETURN
        END

        FUNCTION IFX(X)
        IMPLICIT DOUBLE PRECISION(A-H,O-Z)
        A=SNGL(X-0.001D0)
        B=SNGL(X+0.001D0)
        IF(X.LT.0.0D0) IFX=IFIX(REAL(A))
        IF(X.GT.0.0D0) IFX=IFIX(REAL(B))
        IF(X.EQ.0.0D0) IFX=0
        RETURN
        END

***********************************************************************
! This program is used to convert body-fixed coordinate of a four atoms
! molecule to Cartesian coordinate.
!
!                                 H3
!                                 |\ 
!                H2               | \
!               /                 |D \
!              /                 /    \
!             /theta10          /   r2 \ theta20
!         r1 /_________________/________\_________ phi0
!           /A                C         B\
!          /               R              \
!         /                                \
!        /                                  F
!       H1                                     
!
! Note:
!      r1: the distance of atoms H1 and H2, H1 ---> H2, Unit: Angstrom  
!      r2: the distance of atoms H3 and F,   F ---> H3, Unit: Angstrom
!      R: the distance of the center of mass of molecules H1H2 and H3F
!         A ---> B, Unit: Angstrom
!      theta10: the angle between vectors r1 and R, Unit: Degree
!      theta20: the angle between vectors r2 and R, Unit: Degree
!      phi0: the torsion angle between vectors r1 and r2, Unit: Degree
!      q(3,4):the order of atoms in H1H2H3F, Unit: Angstrom

!      H1,H2 are on the xy-plane, and mid-points(center of mass)A is 
!      the origin of coordinates, B is on x-axis, 
***********************************************************************
        subroutine bf_to_xyz(r1,r2,R,theta10,theta20,phi0,q)
        implicit none
        integer :: i,j
        double precision :: q(3,4),BH3,CH3,DH3
        double precision :: mass(4),theta10,theta20,phi0
        double precision :: r1,r2,R,theta1,theta2,phi
        double precision,parameter:: PI=3.14159265d0

        theta1=theta10*PI/180.d0
        theta2=theta20*PI/180.d0
        phi=phi0*PI/180.d0

        mass(1)=1.00782503d0
        mass(2)=mass(1)
        mass(3)=mass(1)
        mass(4)=18.99840322d0

        q(1,1)=-r1*dcos(theta1)/2.d0
        q(2,1)=-r1*dsin(theta1)/2.d0
        q(3,1)= 0.d0
        q(1,2)= r1*dcos(theta1)/2.d0
        q(2,2)= r1*dsin(theta1)/2.d0
        q(3,2)= 0.d0

        BH3=mass(4)*r2/(mass(3)+mass(4))
        q(1,3)= R+BH3*dcos(theta2)
        q(1,4)= R-(r2-BH3)*dcos(theta2)

        CH3=BH3*dsin(theta2)
        DH3=CH3*dsin(phi)
        q(2,3)=CH3*dcos(phi)
        q(2,4)=-r2*dsin(theta2)*dcos(phi)+q(2,3)

        if (phi<1.d-2)  then
           q(2,3)=BH3*dsin(theta2)
           q(2,4)=-(r2-BH3)*dsin(theta2)
        endif

        q(3,3)= DH3
        q(3,4)=-(r2*dsin(theta2)*dsin(phi)-q(3,3))

        do i=1,3
           do j=1,4
              if (abs(q(i,j))<1.d-8) then
                 q(i,j)=0.d0
              endif
           enddo
        enddo

        return
        end subroutine bf_to_xyz

        subroutine xyz_to_bf(q,r1,r2,R,theta1,theta2,phi)
        implicit none
        double precision :: q(3,4),x(3,4),vec(3,3),A(3),B(3)
        double precision :: mass(4),H1B(3),H2B(3),H3A(3),H3B(3)
        double precision :: n1(3),n2(3),ln1,ln2
        double precision :: r1,r2,R,theta1,theta2,phi
        double precision,parameter:: PI=3.14159265d0
        integer k,m

        x=q ! q(:,4) in order of H1H2H3F
        mass(1)=1.00782503d0
        mass(2)=mass(1)
        mass(3)=mass(1)
        mass(4)=18.99840322d0

        vec(:,1)=x(:,2)-x(:,1) !  H1->H2
        vec(:,2)=x(:,3)-x(:,4) !   F->H3
        r1=dsqrt(dot_product(vec(:,1),vec(:,1)))
        r2=dsqrt(dot_product(vec(:,2),vec(:,2)))

        A(:)=x(:,1)+(mass(2)/(mass(1)+mass(2)))*(x(:,2)-x(:,1))
        B(:)=x(:,4)+(mass(3)/(mass(4)+mass(3)))*(x(:,3)-x(:,4))
        vec(:,3)=B(:)-A(:) !   A->B
        R=dsqrt(dot_product(vec(:,3),vec(:,3)))

        theta1=dot_product(vec(:,1),vec(:,3))/(r1*R)
        theta1=dacos(max(-1.0d0,min(1.0d0,theta1)))
        theta1=theta1/PI*180.d0
        theta2=dot_product(vec(:,2),vec(:,3))/(r2*R)
        theta2=dacos(max(-1.0d0,min(1.0d0,theta2)))
        theta2=theta2/PI*180.d0

        if(abs(theta2-0.d0)<1d-3 .or. abs(theta2-180.d0)<1d-3
     $     .or. abs(theta1-0.d0)<1d-3) then
           phi=0.d0

        else
           H1B(:)=B(:)-x(:,1)
           H2B(:)=B(:)-x(:,2)
           call cross_prod(H1B,H2B,n1)

           H3A=A(:)-x(:,3)
           H3B=B(:)-x(:,3)
           call cross_prod(H3A,H3B,n2)

           ln1=dsqrt(dot_product(n1,n1))
           ln2=dsqrt(dot_product(n2,n2))
           phi=dot_product(n1,n2)/(ln1*ln2)
           phi=dacos(max(-1.0d0,min(1.0d0,phi)))
           phi=(pi-phi)*18.d1/pi

        endif

        return
        end subroutine xyz_to_bf

        subroutine cross_prod(a,b,n)
        implicit none
        double precision :: a(3),b(3),n(3)
        n(1)=a(2)*b(3)-a(3)*b(2)
        n(2)=a(3)*b(1)-a(1)*b(3)
        n(3)=a(1)*b(2)-a(2)*b(1)
        return
        end subroutine cross_prod


C##################################################################
C# SPLINE ROUTINES
C#            Numerical recipes in fortran
C#            Cambrige University Press
C#            York, 2nd edition, 1992.
C##################################################################
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      implicit double precision  (a-h,o-z)
      DIMENSION xa(n),y2a(n),ya(n)
      klo=1
      khi=n
 5    if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 5
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.0d0) write(6,*) 'bad xa input in splint'
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     *2)/6.0d0
      return
      END

C###################################################################
      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
      implicit double precision  (a-h,o-z)
      DIMENSION x(n),y(n),y2(n)
      PARAMETER (NMAX=300)
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
        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
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
12    continue
      return
      END


!        FUNCTION spgndr(l,mm,x) !normalized associated Legendre function
!        implicit real*8 (a-h, o-z)
!        INTEGER l,m,mm

!        m=abs(mm)
!        fact=(2*l+1)/2.d0
!        do i=l-m+1, l+m
!                fact=fact/i
!        enddo
!        fact=sqrt(fact)
!        if(mm.lt.0.and.m/2*2.ne.m) fact=-fact
!        spgndr=fact*plgndr(l,m,x)
!        ENDFUNCTION


!        FUNCTION plgndr(l,m,x)
!        implicit real*8(a-h,o-z)
!        INTEGER l,m
!c        REAL plgndr,x
!c        INTEGER i,ll
!c        REAL fact,pll,pmm,pmmp1,somx2
!        if(m.lt.0.or.m.gt.l.or.abs(x).gt.1.)pause
!     *  'bad arguments in plgndr'
!        pmm=1.
!        if(m.gt.0) then
!         somx2=sqrt((1.-x)*(1.+x))
!          fact=1.
!          do 11 i=1,m
!            pmm=-pmm*fact*somx2
!            fact=fact+2.
!11        continue
!        endif
!        if(l.eq.m) then
!          plgndr=pmm
!        else
!          pmmp1=x*(2*m+1)*pmm
!          if(l.eq.m+1) then
!            plgndr=pmmp1
!          else
!            do 12 ll=m+2,l
!              pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
!              pmm=pmmp1
!              pmmp1=pll
!12          continue
!            plgndr=pll
!          endif
!        endif
!        return
!        ENDFUNCTION
!************************************************************************
!************************************************************************
        subroutine long_range_interaction(Jcb,U_int)
        implicit none
        integer i,j
        real*8,intent(in) :: Jcb(6)
        real*8,intent(out) :: U_int
        real*8,parameter :: pi=dacos(-1.d0)
        real*8,parameter :: tohartree=2.1947463137d5
        real*8 rotA(3,3),rotB(3,3),qA,qB,UA,UB
        real*8 RR,alphA,betaA,gamaA,alphB,betaB,gamaB
        real*8 alphAreg,betaAreg,gamaAreg,alphBreg,betaBreg,gamaBreg
        real*8 rA,rB
        real*8 T1(3),T2(3,3),T3(3,3,3),T4(3,3,3,3),T5(3,3,3,3,3)
        real*8 dmA(3),qmA(3,3),omA(3,3,3),hmA(3,3,3,3),domA(3,3,3,3,3)
        real*8 alphaA(3,3),AA(3,3,3),CA(3,3,3,3),EA(3,3,3,3)
        real*8 dmA_DF(3),qmA_DF(3,3),omA_DF(3,3,3),hmA_DF(3,3,3,3)
        real*8 domA_DF(3,3,3,3,3)
        real*8 alphaA_DF(3,3),AA_DF(3,3,3),CA_DF(3,3,3,3),EA_DF(3,3,3,3)
        real*8 dmB(3),qmB(3,3),omB(3,3,3),hmB(3,3,3,3),domB(3,3,3,3,3)
        real*8 alphaB(3,3),AB(3,3,3),CB(3,3,3,3),EB(3,3,3,3)
        real*8 dmB_DF(3),qmB_DF(3,3),omB_DF(3,3,3),hmB_DF(3,3,3,3)
        real*8 domB_DF(3,3,3,3,3)
        real*8 alphaB_DF(3,3),AB_DF(3,3,3),CB_DF(3,3,3,3),EB_DF(3,3,3,3)
        real*8 U_el,U_ind_A,U_ind_B,U_disp

        RR=Jcb(1)
        rA=Jcb(2)
        rB=Jcb(3)
        call mult_polar_A(rA,UA,qA,dmA,qmA,omA,hmA,domA,alphaA,AA,CA,EA)
        call mult_polar_B(rB,UB,qB,dmB,qmB,omB,hmB,domB,alphaB,AB,CB,EB)

        alphA=0.d0
        betaA=Jcb(4)
        gamaA=0.d0
        alphB=Jcb(6)
        betaB=Jcb(5)
        gamaB=0.d0

        alphAreg=alphA*pi/18d1
        betaAreg=betaA*pi/18d1
        gamaAreg=gamaA*pi/18d1
        alphBreg=alphB*pi/18d1
        betaBreg=betaB*pi/18d1
        gamaBreg=gamaB*pi/18d1        

        call eulerrot(rotA,alphAreg,betaAreg,gamaAreg,1)
        call eulerrot(rotB,alphBreg,betaBreg,gamaBreg,1)
        call trans_elect(rotA,dmA,qmA,omA,hmA,domA,alphaA,AA,CA,EA,
     &                   dmA_DF,qmA_DF,omA_DF,hmA_DF,domA_DF,
     &                   alphaA_DF,AA_DF,CA_DF,EA_DF)
        call trans_elect(rotB,dmB,qmB,omB,hmB,domB,alphaB,AB,CB,EB,
     &                   dmB_DF,qmB_DF,omB_DF,hmB_DF,domB_DF,
     &                   alphaB_DF,AB_DF,CB_DF,EB_DF)
        call T(RR,T1,T2,T3,T4,T5)

        call lr_el(RR,T1,T2,T3,T4,T5,qA,dmA_DF,qmA_DF,omA_DF,hmA_DF,
     &             domA_DF,qB,dmB_DF,qmB_DF,omB_DF,hmB_DF,domB_DF,
     &             U_el)
        call lr_ind(T1,T2,T3,T4,T5,dmA_DF,qmA_DF,omA_DF,hmA_DF,
     &              alphaB_DF,AB_DF,CB_DF,EB_DF,U_ind_B,0)!0 for U_ind_B
        call lr_ind(T1,T2,T3,T4,T5,dmB_DF,qmB_DF,omB_DF,hmB_DF,
     &              alphaA_DF,AA_DF,CA_DF,EA_DF,U_ind_A,1)!1 for U_ind_A
        call lr_disp(RR,T2,T3,T4,alphaA_DF,AA_DF,CA_DF,EA_DF,alphaB_DF,
     &               AB_DF,CB_DF,EB_DF,UA,UB,U_disp)

        U_int=U_el+U_ind_A+U_ind_B+U_disp

        endsubroutine


        subroutine mult_polar_A(rA,UA,qA,dmA,qmA,omA,hmA,domA,alphaA,AA,
     &                          CA,EA)
        implicit none
        real*8,intent(in) :: rA
        real*8 tmp0
        real*8 UA,qA
        real*8 dmA(3),qmA(3,3),omA(3,3,3),hmA(3,3,3,3),domA(3,3,3,3,3)
        real*8 alphaA(3,3),AA(3,3,3),CA(3,3,3,3),EA(3,3,3,3)

        UA=15.43d0/27.21140d0

        qA=0.d0;dmA=0.d0;qmA=0.d0;omA=0.d0;hmA=0.d0;domA=0.d0
        alphaA=0.d0;AA=0.d0;CA=0.d0;EA=0.d0

        qmA(3,3)=-0.0927d0*rA**3+0.4234d0*rA**2-0.1075d0*rA+0.0329d0 
        hmA(3,3,3,3)=0.0205d0*rA**3+0.5526d0*rA**2-0.8582d0*rA+0.3395d0 
        alphaA(1,1)=-0.3406d0*rA**3+1.6100d0*rA**2+0.7269d0*rA+1.3442d0 
        alphaA(3,3)=-0.7497d0*rA**3+4.9608d0*rA**2-2.7281d0*rA+2.5736d0 
        CA(3,3,3,3)=0.7969d0*rA**3+0.7587d0*rA**2+1.1578d0*rA+0.6389d0 
        CA(1,2,1,2)=-0.3084d0*rA**3+1.4463d0*rA**2+0.1250d0*rA+0.6412d0 
        CA(1,3,1,3)=0.1110d0*rA**3+1.7933d0*rA**2-0.3610d0*rA+0.8836d0 
        EA(3,3,3,3)=4.4161d0*rA**3-8.2773d0*rA**2+7.1058d0*rA-2.0101d0 
        EA(1,1,3,3)=1.5269d0*rA**3-1.9411d0*rA**2+1.4980d0*rA-0.3730d0 

        qmA(2,2)=-0.5d0*qmA(3,3)
        qmA(1,1)=qmA(2,2)

        hmA(1,1,3,3)=-0.5d0*hmA(3,3,3,3)
        hmA(2,2,3,3)=hmA(1,1,3,3)
        hmA(1,1,1,1)=3.d0*hmA(3,3,3,3)/8.d0
        hmA(2,2,2,2)=hmA(1,1,1,1)
        hmA(1,1,2,2)=-hmA(1,1,3,3)-hmA(1,1,1,1)

        alphaA(2,2)=alphaA(1,1)

        CA(1,1,3,3)=-0.5d0*CA(3,3,3,3)
        CA(2,2,3,3)=CA(1,1,3,3)
        CA(1,1,1,1)=CA(3,3,3,3)/4.d0+CA(1,2,1,2)
        CA(2,2,2,2)=CA(1,1,1,1)
        CA(1,1,2,2)=-CA(1,1,3,3)-CA(1,1,1,1)
        CA(2,3,2,3)=CA(1,3,1,3)

        EA(3,2,2,3)=-0.5d0*EA(3,3,3,3)
        EA(3,1,1,3)=EA(3,2,2,3)
        EA(1,1,1,1)=-EA(1,1,3,3)*3.d0/4.d0
        EA(1,1,2,2)=-EA(1,1,1,1)-EA(1,1,3,3)
        EA(2,2,3,3)=EA(1,1,3,3)
        EA(2,2,2,2)=EA(1,1,1,1)
        EA(2,1,1,2)=-EA(2,2,3,3)-EA(2,2,2,2)

        call get_all_ele_hm(hmA)
        call get_all_ele_C(CA)
        call get_all_ele_E(EA)

        endsubroutine


        subroutine mult_polar_B(rB,UB,qB,dmB,qmB,omB,hmB,domB,alphaB,AB,
     &                          CB,EB)
        implicit none
        real*8,intent(in) :: rB
        real*8 tmp0
        real*8 UB,qB
        real*8 dmB(3),qmB(3,3),omB(3,3,3),hmB(3,3,3,3),domB(3,3,3,3,3)
        real*8 alphaB(3,3),AB(3,3,3),CB(3,3,3,3),EB(3,3,3,3)

        UB=16.06d0/27.21140d0

        qB=0.d0;dmB=0.d0;qmB=0.d0;omB=0.d0;hmB=0.d0;domB=0.d0
        alphaB=0.d0;AB=0.d0;CB=0.d0;EB=0.d0

        dmB(3)=-0.0659d0*rB**3+0.3215d0*rB**2-0.1959d0*rB+0.4275d0 
        qmB(3,3)=-0.2364d0*rB**3+1.5484d0*rB**2-1.6122d0*rB+1.0968d0 
        omB(3,3,3)=-0.2114d0*rB**3+3.0239d0*rB**2-4.1869d0*rB+1.8127d0 
        hmB(3,3,3,3)=1.4298d0*rB**3+0.3631d0*rB**2-3.2991d0*rB+2.0442d0 
        alphaB(1,1)=-0.0219d0*rB**3+0.1828d0*rB**2+0.9641d0*rB+2.9698d0 
        alphaB(3,3)=0.7406d0*rB**3-0.7544d0*rB**2+1.6953d0*rB+1.7847d0 
        AB(1,1,3)=0.3786d0*rB**3-0.4837d0*rB**2+0.5845d0*rB-0.2047d0 
        AB(3,3,3)=5.0793d0*rB**3-15.1237d0*rB**2+18.0467d0*rB-7.9295d0 
        CB(3,3,3,3)=7.9258d0*rB**3-28.1790d0*rB**2+36.5186d0*rB
     &-12.2719d0 
        CB(1,2,1,2)=-0.2302d0*rB**3+1.2187d0*rB**2-0.8735d0*rB+2.8456d0 
        CB(1,3,1,3)=1.0548d0*rB**3-2.9948d0*rB**2+4.3884d0*rB+0.6658d0 
        EB(3,3,3,3)=23.4025d0*rB**3-82.3799d0*rB**2+101.4015d0*rB
     &-44.4284d0 
        EB(1,1,3,3)=2.9429d0*rB**3-8.1027d0*rB**2+9.3275d0*rB-5.6484d0 

        qmB(2,2)=-0.5d0*qmB(3,3)
        qmB(1,1)=qmB(2,2)

        omB(1,1,3)=-0.5d0*omB(3,3,3)
        omB(2,2,3)=omB(1,1,3)

        hmB(1,1,3,3)=-0.5d0*hmB(3,3,3,3)
        hmB(2,2,3,3)=hmB(1,1,3,3)
        hmB(1,1,1,1)=3.d0*hmB(3,3,3,3)/8.d0
        hmB(2,2,2,2)=hmB(1,1,1,1)
        hmB(1,1,2,2)=-hmB(1,1,3,3)-hmB(1,1,1,1)

        alphaB(2,2)=alphaB(1,1)

        AB(3,1,1)=-0.5d0*AB(3,3,3)
        AB(3,2,2)=AB(3,1,1)
        AB(2,2,3)=AB(1,1,3)
        AB(1,3,1)=AB(1,1,3)
        AB(2,3,2)=AB(2,2,3)

        CB(1,1,3,3)=-0.5d0*CB(3,3,3,3)
        CB(2,2,3,3)=CB(1,1,3,3)
        CB(1,1,1,1)=CB(3,3,3,3)/4.d0+CB(1,2,1,2)
        CB(2,2,2,2)=CB(1,1,1,1)
        CB(1,1,2,2)=-CB(1,1,3,3)-CB(1,1,1,1)
        CB(2,3,2,3)=CB(1,3,1,3)

        EB(3,2,2,3)=-0.5d0*EB(3,3,3,3)
        EB(3,1,1,3)=EB(3,2,2,3)
        EB(1,1,1,1)=-EB(1,1,3,3)*3.d0/4.d0
        EB(1,1,2,2)=-EB(1,1,1,1)-EB(1,1,3,3)
        EB(2,2,3,3)=EB(1,1,3,3)
        EB(2,2,2,2)=EB(1,1,1,1)
        EB(2,1,1,2)=-EB(2,2,3,3)-EB(2,2,2,2)

        call get_all_ele_om(omB)
        call get_all_ele_hm(hmB)
        call get_all_ele_C(CB)
        call get_all_ele_E(EB)

        endsubroutine


        subroutine get_all_ele_om(om)
        implicit none
        integer i,j,k
        real*8 om(3,3,3)
        real*8 tmp

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           if(abs(om(i,j,k))<1.d-8)then
              cycle
           else
              tmp=om(i,j,k)
              om(i,k,j)=tmp
              om(j,i,k)=tmp
              om(j,k,i)=tmp
              om(k,j,i)=tmp
              om(k,i,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo

        endsubroutine


        subroutine get_all_ele_hm(hm)
        implicit none
        integer i,j,k,l
        real*8 hm(3,3,3,3)
        real*8 tmp

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(abs(hm(i,j,k,l))<1.d-8)then
              cycle
           else
              tmp=hm(i,j,k,l)
              hm(i,j,l,k)=tmp
              hm(i,k,j,l)=tmp
              hm(i,k,l,j)=tmp
              hm(i,l,j,k)=tmp
              hm(i,l,k,j)=tmp
              hm(j,i,k,l)=tmp
              hm(j,i,l,k)=tmp
              hm(j,k,i,l)=tmp
              hm(j,k,l,i)=tmp
              hm(j,l,i,k)=tmp
              hm(j,l,k,i)=tmp
              hm(k,i,j,l)=tmp
              hm(k,i,l,j)=tmp
              hm(k,j,i,l)=tmp
              hm(k,j,l,i)=tmp
              hm(k,l,i,j)=tmp
              hm(k,l,j,i)=tmp
              hm(l,i,j,k)=tmp
              hm(l,i,k,j)=tmp
              hm(l,j,i,k)=tmp
              hm(l,j,k,i)=tmp
              hm(l,k,i,j)=tmp
              hm(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_all_ele_C(C)
        implicit none
        integer i,j,k,l
        real*8 C(3,3,3,3)
        real*8 tmp

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(abs(C(i,j,k,l))<1.d-8)then
              cycle
           else
              tmp=C(i,j,k,l)
              C(i,j,l,k)=tmp
              C(j,i,k,l)=tmp
              C(j,i,l,k)=tmp
              C(k,l,i,j)=tmp
              C(k,l,j,i)=tmp
              C(l,k,i,j)=tmp
              C(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_all_ele_E(E)
        implicit none
        integer i,j,k,l
        real*8 E(3,3,3,3)
        real*8 tmp

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(abs(E(i,j,k,l))<1.d-8)then
              cycle
           else
              tmp=E(i,j,k,l)
              E(i,j,l,k)=tmp
              E(i,k,j,l)=tmp
              E(i,k,l,j)=tmp
              E(i,l,j,k)=tmp
              E(i,l,k,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine eulerrot(rot,alpha,beta,gamma,transpose)
        implicit real*8 (a-h,o-z)
        double precision rot(3,3)
        integer transpose

        cosa=dcos(alpha)
        sina=dsin(alpha)
        cosb=dcos(beta)
        sinb=dsin(beta)
        cosc=dcos(gamma)
        sinc=dsin(gamma)

        if( transpose .eq. 0 ) then
           rot(1,1) = cosb*cosa*cosc-sina*sinc
           rot(1,2) = cosb*sina*cosc+cosa*sinc
           rot(1,3) = -sinb*cosc
           rot(2,1) = -cosb*cosa*sinc-sina*cosc
           rot(2,2) = -cosb*sina*sinc+cosa*cosc
           rot(2,3) = sinb*sinc
           rot(3,1) = sinb*cosa
           rot(3,2) = sinb*sina
           rot(3,3) = cosb
        else
           rot(1,1) = cosb*cosa*cosc-sina*sinc
           rot(2,1) = cosb*sina*cosc+cosa*sinc
           rot(3,1) = -sinb*cosc
           rot(1,2) = -cosb*cosa*sinc-sina*cosc
           rot(2,2) = -cosb*sina*sinc+cosa*cosc
           rot(3,2) = sinb*sinc
           rot(1,3) = sinb*cosa
           rot(2,3) = sinb*sina
           rot(3,3) = cosb
         endif
         return
         endsubroutine


        subroutine T(R,T1,T2,T3,T4,T5)
        implicit none
        integer i,j,k,l,m
        real*8 R,RR(3),delta(3,3)
        real*8 T1(3),T2(3,3),T3(3,3,3),T4(3,3,3,3),T5(3,3,3,3,3)

        delta=0.d0
        do i=1,3,1
           delta(i,i)=1.d0
        enddo

        RR=0.d0
        RR(3)=R

        T1=0.d0
        do i=1,3,1
           T1(i)=-RR(i)/R**3.d0
        enddo

        T2=0.d0
        do i=1,3,1
        do j=1,3,1
           T2(i,j)=(3.d0*RR(i)*RR(j)-R**2.d0
     &              *delta(i,j))/R**5.d0
        enddo;enddo

        T3=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           T3(i,j,k)=-(15.d0*RR(i)*RR(j)*RR(k)-3.d0*R**2.d0
     &         *(RR(i)*delta(j,k)+RR(j)*delta(i,k)+RR(k)*delta(i,j)))
     &         /R**7.d0
        enddo;enddo;enddo

        T4=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           T4(i,j,k,l)=(105.d0*RR(i)*RR(j)*RR(k)*RR(l)-15.d0*R**2.d0
     &         *(RR(i)*RR(j)*delta(k,l)+RR(i)*RR(k)*delta(j,l)
     &         +RR(i)*RR(l)*delta(j,k)+RR(j)*RR(k)*delta(i,l)
     &         +RR(j)*RR(l)*delta(i,k)+RR(k)*RR(l)*delta(i,j))
     &         +3.d0*R**4.d0*(delta(i,j)*delta(k,l)+
     &         delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k)))/R**9.d0
        enddo;enddo;enddo;enddo

        T5=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           T5(i,j,k,l,m)=(-945.d0*RR(i)*RR(j)*RR(k)*RR(l)*RR(m)+
     &         105.d0*R**2.d0*(RR(i)*RR(j)*RR(k)*delta(l,m)+
     &         RR(i)*RR(j)*RR(l)*delta(k,m)+RR(i)*RR(j)*RR(m)*delta(k,l)
     &        +RR(i)*RR(k)*RR(l)*delta(j,m)+RR(i)*RR(k)*RR(m)*delta(j,l)
     &        +RR(i)*RR(l)*RR(m)*delta(j,k)+RR(j)*RR(k)*RR(l)*delta(i,m)
     &        +RR(j)*RR(k)*RR(m)*delta(i,l)+RR(j)*RR(l)*RR(m)*delta(i,k)
     &        +RR(k)*RR(l)*RR(m)*delta(i,j))-15.d0*R**4.d0*(
     &         RR(i)*delta(j,k)*delta(l,m)+RR(i)*delta(j,l)*delta(k,m)
     &        +RR(i)*delta(j,m)*delta(k,l)+RR(j)*delta(i,k)*delta(l,m)
     &        +RR(j)*delta(i,l)*delta(k,m)+RR(j)*delta(i,m)*delta(k,l)
     &        +RR(k)*delta(i,j)*delta(l,m)+RR(k)*delta(i,l)*delta(j,m)
     &        +RR(k)*delta(i,m)*delta(j,l)+RR(l)*delta(i,j)*delta(k,m)
     &        +RR(l)*delta(i,k)*delta(j,m)+RR(l)*delta(i,m)*delta(j,k)
     &        +RR(m)*delta(i,j)*delta(k,l)+RR(m)*delta(i,k)*delta(j,l)
     &        +RR(m)*delta(i,l)*delta(j,k)))/R**11.d0
        enddo;enddo;enddo;enddo;enddo

        return
        endsubroutine


        subroutine trans_elect(rot,dm,qm,om,hm,dom,alpha,A,C,E,
     &                         dm_DF,qm_DF,om_DF,hm_DF,dom_DF,
     &                         alpha_DF,A_DF,C_DF,E_DF)
        implicit none
        integer i,j,k,l,m,n,o,p,s,t
        real*8 rot(3,3)
        real*8 dm(3),qm(3,3),alpha(3,3),om(3,3,3),A(3,3,3)
        real*8 hm(3,3,3,3),C(3,3,3,3),E(3,3,3,3),dom(3,3,3,3,3)
        real*8 dm_DF(3),qm_DF(3,3),alpha_DF(3,3),om_DF(3,3,3)
        real*8 A_DF(3,3,3),hm_DF(3,3,3,3),C_DF(3,3,3,3),E_DF(3,3,3,3)
        real*8 dom_DF(3,3,3,3,3)

        dm_DF=0.d0;qm_DF=0.d0;alpha_DF=0.d0;om_DF=0.d0;A_DF=0.d0
        hm_DF=0.d0;C_DF=0.d0;E_DF=0.d0;dom_DF=0.d0

        do i=1,3,1
        do j=1,3,1
           dm_DF(i)=dm_DF(i)+rot(i,j)*dm(j)
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           qm_DF(i,j)=qm_DF(i,j)+rot(i,k)*rot(j,l)*qm(k,l)
           alpha_DF(i,j)=alpha_DF(i,j)+rot(i,k)*rot(j,l)*alpha(k,l)
        enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           om_DF(i,j,k)=om_DF(i,j,k)+rot(i,l)*rot(j,m)*rot(k,n)*
     &                  om(l,m,n)
           A_DF(i,j,k)=A_DF(i,j,k)+rot(i,l)*rot(j,m)*rot(k,n)*A(l,m,n)
        enddo;enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
        do o=1,3,1
        do p=1,3,1
           hm_DF(i,j,k,l)=hm_DF(i,j,k,l)+rot(i,m)*rot(j,n)*rot(k,o)
     &                    *rot(l,p)*hm(m,n,o,p)
           C_DF(i,j,k,l)=C_DF(i,j,k,l)+rot(i,m)*rot(j,n)*rot(k,o)
     &                   *rot(l,p)*C(m,n,o,p)
           E_DF(i,j,k,l)=E_DF(i,j,k,l)+rot(i,m)*rot(j,n)*rot(k,o)
     &                   *rot(l,p)*E(m,n,o,p)
        enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
        do o=1,3,1
        do p=1,3,1
        do s=1,3,1
        do t=1,3,1
           dom_DF(i,j,k,l,m)=dom_DF(i,j,k,l,m)+rot(i,n)*rot(j,o)
     &                       *rot(k,p)*rot(l,s)*rot(m,t)*dom(n,o,p,s,t)
        enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo

        endsubroutine


        subroutine lr_el(R,T1,T2,T3,T4,T5,qA,dmA,qmA,omA,hmA,domA,qB,
     &                   dmB,qmB,omB,hmB,domB,U_el)
        implicit none
        integer i,j,k,l,m
        real*8 R,T1(3),T2(3,3),T3(3,3,3),T4(3,3,3,3),T5(3,3,3,3,3)
        real*8 qA,dmA(3),qmA(3,3),omA(3,3,3),hmA(3,3,3,3)
        real*8 qB,dmB(3),qmB(3,3),omB(3,3,3),hmB(3,3,3,3)
        real*8 domA(3,3,3,3,3),domB(3,3,3,3,3)
        real*8 term(21),U_el

        U_el=0.d0
        term=0.d0

        term(1)=qA*qB/R

        do i=1,3,1
           term(2)=term(2)+T1(i)*qA*dmB(i)
           term(3)=term(3)-T1(i)*dmA(i)*qB
        enddo

        do i=1,3,1
        do j=1,3,1
           term(4)=term(4)+T2(i,j)*qA*qmB(i,j)/3.d0
           term(5)=term(5)-T2(i,j)*dmA(i)*dmB(j)
           term(6)=term(6)+T2(i,j)*qmA(i,j)*qB/3.d0
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           term(7)=term(7)+T3(i,j,k)*qA*omB(i,j,k)/15.d0
           term(8)=term(8)-T3(i,j,k)*dmA(i)*qmB(j,k)/3.d0
           term(9)=term(9)+T3(i,j,k)*qmA(i,j)*dmB(k)/3.d0
           term(10)=term(10)-T3(i,j,k)*omA(i,j,k)*qB/15.d0
        enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(11)=term(11)+T4(i,j,k,l)*qA*hmB(i,j,k,l)/105.d0
           term(12)=term(12)-T4(i,j,k,l)*dmA(i)*omB(j,k,l)/15.d0
           term(13)=term(13)+T4(i,j,k,l)*qmA(i,j)*qmB(k,l)/9.d0
           term(14)=term(14)-T4(i,j,k,l)*omA(i,j,k)*dmB(l)/15.d0
           term(15)=term(15)+T4(i,j,k,l)*hmA(i,j,k,l)*qB/105.d0
        enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(16)=term(16)+T5(i,j,k,l,m)*qA*domB(i,j,k,l,m)/945.d0
           term(17)=term(17)-T5(i,j,k,l,m)*dmA(i)*hmB(j,k,l,m)/105.d0
           term(18)=term(18)+T5(i,j,k,l,m)*qmA(i,j)*omB(k,l,m)/45.d0
           term(19)=term(19)-T5(i,j,k,l,m)*omA(i,j,k)*qmB(l,m)/45.d0
           term(20)=term(20)+T5(i,j,k,l,m)*hmA(i,j,k,l)*dmB(m)/105.d0
           term(21)=term(21)-T5(i,j,k,l,m)*domA(i,j,k,l,m)*qB/945.d0
        enddo;enddo;enddo;enddo;enddo

        term(:)=term(:)*2.1947463137d5
        U_el=sum(term(1:21))
!        write(*,'(7f15.5)')term(1),sum(term(2:3)),sum(term(4:6)),
!     &            sum(term(7:10)),sum(term(11:15)),sum(term(16:21)),U_el

        return
        endsubroutine



        subroutine lr_ind(T1,T2,T3,T4,T5,dm1,qm1,om1,hm1,alpha2,A2,C2,
     &                    E2,U_ind,judge)
        implicit none
        integer i,j,k,l,m,n,judge
        real*8 T1(3),T2(3,3),T3(3,3,3),T4(3,3,3,3),T5(3,3,3,3,3)
        real*8 q1,dm1(3),qm1(3,3),om1(3,3,3),hm1(3,3,3,3)
        real*8 alpha2(3,3),A2(3,3,3),C2(3,3,3,3),E2(3,3,3,3)
        real*8 term(29),U_ind

        U_ind=0.d0
        term=0.d0

        do i=1,3,1
        do j=1,3,1
           term(1)=term(1)-T1(i)*T1(j)*q1*q1*alpha2(i,j)/2.d0
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           term(2)=term(2)+T1(i)*T2(j,k)*q1*dm1(k)*alpha2(i,j)
           term(3)=term(3)-T1(i)*T2(j,k)*q1*q1*A2(i,j,k)/3.d0
        enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(4)=term(4)-T1(i)*T3(j,k,l)*q1*qm1(k,l)*alpha2(i,j)/3.d0
           term(5)=term(5)-T2(i,j)*T2(k,l)*dm1(j)*dm1(l)
     &             *alpha2(i,k)/2.d0
           term(6)=term(6)+T1(i)*T3(j,k,l)*q1*dm1(l)*A2(i,j,k)/3.d0
           term(7)=term(7)+T2(i,j)*T2(k,l)*dm1(j)*q1*A2(i,k,l)/3.d0
           term(8)=term(8)-T2(i,j)*T2(k,l)*q1*q1*C2(i,j,k,l)/6.d0
           term(9)=term(9)-T1(i)*T3(j,k,l)*q1*q1*E2(i,j,k,l)/15.d0
        enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(10)=term(10)+T1(i)*T4(j,k,l,m)*q1*om1(k,l,m)
     &              *alpha2(i,j)/15.d0
           term(11)=term(11)+T2(i,j)*T3(k,l,m)*dm1(j)*qm1(l,m)
     &              *alpha2(i,k)/3.d0
           term(12)=term(12)-T1(i)*T4(j,k,l,m)*q1*qm1(l,m)
     &              *A2(i,j,k)/9.d0
           term(13)=term(13)-T2(i,j)*T3(k,l,m)*dm1(j)*dm1(m)
     &              *A2(i,k,l)/3.d0
           term(14)=term(14)-T3(i,j,k)*T2(l,m)*qm1(j,k)*q1
     &              *A2(i,l,m)/9.d0
           term(15)=term(15)+T2(i,j)*T3(k,l,m)*q1*dm1(m)
     &              *C2(i,j,k,l)/3.d0
           term(16)=term(16)+T1(i)*T4(j,k,l,m)*q1*dm1(m)
     &              *E2(i,j,k,l)/15.d0
           term(17)=term(17)+T2(i,j)*T3(k,l,m)*dm1(j)*q1
     &              *E2(i,k,l,m)/15.d0
        enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           term(18)=term(18)-T1(i)*T5(j,k,l,m,n)*q1*hm1(k,l,m,n)
     &              *alpha2(i,j)/105.d0
           term(19)=term(19)-T2(i,j)*T4(k,l,m,n)*dm1(j)*om1(l,m,n)
     &              *alpha2(i,k)/15.d0
           term(20)=term(20)-T3(i,j,k)*T3(l,m,n)*qm1(j,k)*qm1(m,n)
     &              *alpha2(i,l)/18.d0
           term(21)=term(21)+T1(i)*T5(j,k,l,m,n)*q1*om1(l,m,n)
     &              *A2(i,j,k)/45.d0
           term(22)=term(22)+T2(i,j)*T4(k,l,m,n)*dm1(j)*qm1(m,n)
     &              *A2(i,k,l)/9.d0
           term(23)=term(23)+T3(i,j,k)*T3(l,m,n)*qm1(j,k)*dm1(n)
     &              *A2(i,l,m)/9.d0
           term(24)=term(24)+T4(i,j,k,l)*T2(m,n)*om1(j,k,l)*q1
     &              *A2(i,m,n)/45.d0
           term(25)=term(25)-T2(i,j)*T4(k,l,m,n)*q1*qm1(m,n)
     &              *C2(i,j,k,l)/9.d0
           term(26)=term(26)-T3(i,j,k)*T3(l,m,n)*dm1(k)*dm1(n)
     &              *C2(i,j,l,m)/6.d0
           term(27)=term(27)-T1(i)*T5(j,k,l,m,n)*q1*qm1(m,n)
     &              *E2(i,j,k,l)/45.d0
           term(28)=term(28)-T2(i,j)*T4(k,l,m,n)*dm1(j)*dm1(n)
     &              *E2(i,k,l,m)/15.d0
           term(29)=term(29)-T3(i,j,k)*T3(l,m,n)*qm1(j,k)*q1
     &              *E2(i,l,m,n)/45.d0
        enddo;enddo;enddo;enddo;enddo;enddo

        term(:)=term(:)*2.1947463137d5
        U_ind=term(1)+(-1.d0)**real(judge)*sum(term(2:3))+sum(term(4:9))
     &        +(-1.d0)**real(judge)*sum(term(10:17))+sum(term(18:29))
        !write(82,'(6f15.5)')term(1),sum(term(2:3)),sum(term(4:9)),
!     &            sum(term(10:17)),sum(term(18:29)),U_ind
        return
        endsubroutine


        subroutine lr_disp(RR,T2,T3,T4,alphaA,AA,CA,EA,
     &             alphaB,AB,CB,EB,UA,UB,U_disp)
        implicit none
        integer i,j,k,l,m,n
        real*8 RR,T2(3,3),T3(3,3,3),T4(3,3,3,3)
        real*8 alphaA(3,3),AA(3,3,3),CA(3,3,3,3),EA(3,3,3,3)
        real*8 alphaB(3,3),AB(3,3,3),CB(3,3,3,3),EB(3,3,3,3)
        real*8 term(9),UA,UB,U_disp

        U_disp=0.d0
        term=0.d0

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(1)=term(1)-T2(i,j)*T2(k,l)*alphaA(i,k)*alphaB(j,l)
     &             *UA*UB/(4.d0*(UA+UB))
        enddo;enddo;enddo;enddo

!        term(1)=-12.67315d0/RR**6.d0

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(2)=term(2)-T2(i,j)*T3(k,l,m)*alphaA(i,k)*AB(j,l,m)
     &             *UA*UB/(6.d0*(UA+UB))
           term(3)=term(3)+T2(i,j)*T3(k,l,m)*AA(i,k,l)*alphaB(j,m)
     &             *UA*UB/(6.d0*(UA+UB))
        enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           term(4)=term(4)+T2(i,j)*T4(k,l,m,n)*AA(i,k,l)*AB(j,m,n)
     &             *UA*UB/(18.d0*(UA+UB))
           term(5)=term(5)+T3(i,j,k)*T3(l,m,n)*AA(i,l,m)*AB(n,j,k)
     &             *UA*UB/(18.d0*(UA+UB))
           term(6)=term(6)-T3(i,j,k)*T3(l,m,n)*alphaA(i,l)*CB(j,k,m,n)
     &             *UA*UB/(12.d0*(UA+UB))
           term(7)=term(7)-T3(i,j,k)*T3(l,m,n)*CA(i,j,l,m)*alphaB(k,n)
     &             *UA*UB/(12.d0*(UA+UB))
           term(8)=term(8)-T2(i,j)*T4(k,l,m,n)*alphaA(i,k)*EB(j,l,m,n)
     &             *UA*UB/(30.d0*(UA+UB))
           term(9)=term(9)-T2(i,j)*T4(k,l,m,n)*EA(i,k,l,m)*alphaB(j,n)
     &             *UA*UB/(30.d0*(UA+UB))
        enddo;enddo;enddo;enddo;enddo;enddo

        term(:)=term(:)*2.1947463137d5
        U_disp=sum(term(1:9))

        !write(82,'(4f15.5)')term(1),sum(term(2:3)),sum(term(4:9)),U_disp

        return
        endsubroutine

