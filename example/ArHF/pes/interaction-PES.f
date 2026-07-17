        subroutine interaction_PES(rHF,RR,th,V)
        implicit none

        real*8 r1,r2,cth,vfun
        real*8 rHF,RR,th,v

        r1=rHF
        r2=RR
        cth=th
        call arhfpes(r1,r2,cth,vfun)
        V=vfun

        endsubroutine

c=====================================================================
        module nnparam
        implicit none
        integer ninput,noutput,nhid,nlayer,nscale,ifunc,nwe,nodemax
        integer, allocatable::nodes(:)
        real*8, allocatable::weight(:,:,:),bias(:,:)
        real*8, allocatable::pdel(:),pavg(:)
        end module nnparam
c=====================================================================

        subroutine arhfpes(rhf,rr,th,vx)
        use nnparam
        implicit none
        real*8,parameter :: pi=dacos(-1d0)
        real*8,parameter :: bohr=0.529177249d0
        real*8,parameter :: tocm=219474.63067d0
        real*8,intent(in) :: rhf,rr,th ! in bohr and  degree
        real*8,intent(out):: vx ! hartree
        real*8 x(3),vnn,vlr,DR,cth
        integer,save :: init
        data init/0/

        if(init.eq.0)then
        call nnpes_init
        call lrpes_init
        init=1
        endif

        x(1)=rhf ! bohr
        x(2)=th  ! degree
        x(3)=rr*bohr  ! bohr to angstrom
        call arhfpesNN(x,vnn)
        vnn=vnn/tocm

        call arhflr(rhf,rr,th,vlr) ! all in a.u, th in degree

        DR=1d0/(1d0+dexp(-3d0*(x(3)-8d0)))
        vx=(1d0-DR)*vnn+DR*vlr
c        write(123,'(3f9.2,3f12.5)')rhf,rr,th,vnn*tocm,vlr*tocm,vx*tocm

        return
        endsubroutine



        subroutine nnpes_init
        use nnparam
        implicit none
        integer i,ihid,iwe,inode1,inode2,ilay1,ilay2
        character f1*80
        open(111,file='weights.txt-23')
        open(222,file='biases.txt-23')
        read(111,*)ninput,nhid,noutput
        nscale=ninput+noutput
        nlayer=nhid+2
        allocate(nodes(nlayer),pdel(nscale),pavg(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(111,*)(nodes(ihid),ihid=2,nhid+1)
        nodemax=0
        do i=1,nlayer
        nodemax=max(nodemax,nodes(i))
        enddo
       allocate(weight(nodemax,nodemax,2:nlayer),bias(nodemax,2:nlayer))
        read(111,*)ifunc,nwe
!-->....ifunc hence controls the type of transfer function used for
!hidden layers
!-->....At this time, only an equivalent transfer function can be used
!for all hidden layers
!-->....and the pure linear function is always applid to the output layer.
!-->....see function tranfun() for details
        read(111,*)(pdel(i),i=1,nscale)
        read(111,*)(pavg(i),i=1,nscale)
        iwe=0
        do ilay1=2,nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        do inode2=1,nodes(ilay2)
        iwe=iwe+1
        enddo
        iwe=iwe+1
        enddo
        enddo
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif

        do ilay1=2,nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        do inode2=1,nodes(ilay2)
        read(111,*)weight(inode2,inode1,ilay1)
        enddo
        read(222,*)bias(inode1,ilay1)
        enddo
        enddo
        close(111)
        close(222)
        end subroutine

        subroutine arhfpesNN(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun
!-->....set up the normalized input layer
        do i=1,ninput
        y(i,1)=(x(i)-pavg(i))/pdel(i)
        enddo

!-->....evaluate the hidden layer
        do ilay1=2,nlayer-1
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        y(inode1,ilay1)=bias(inode1,ilay1)
        do inode2=1,nodes(ilay2)
        y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     1   *weight(inode2,inode1,ilay1)
        enddo
        y(inode1,ilay1)=tranfun(y(inode1,ilay1),ifunc)
        enddo
        enddo

!-->....now evaluate the output
        ilay1=nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        y(inode1,ilay1)=bias(inode1,ilay1)
        do inode2=1,nodes(ilay2)
        y(inode1,ilay1)=y(inode1,ilay1)+y(inode2,ilay2)
     1   *weight(inode2,inode1,ilay1)
        enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdel(nscale)+pavg(nscale)
        end subroutine

        function tranfun(x,ifunc)
        implicit none
        integer ifunc
        real*8 tranfun,x
!    ifunc=1, transfer function is hyperbolic tangent function, 'tansig'
!    ifunc=2, transfer function is log sigmoid function, 'logsig'
!    ifunc=3, transfer function is pure linear function, 'purelin',
!  which is imposed to the output layer by default
        if (ifunc.eq.1) then
        tranfun=tanh(x)
        else if (ifunc.eq.2) then
        tranfun=1d0/(1d0+exp(-x))
        else if (ifunc.eq.3) then
        tranfun=x
        endif
        end function

        
        real*8 function vr(r)   !only for the potential of HF
        implicit real*8(a-z)
        toai=0.529177249d0
        tocm=219474.64d0
        re=0.9168d0/toai
        De=49361.6/tocm
        a1=2.23729d0
        a2=1.12367d0
        a3=0.568735d0
        a4=0.00918165d0
        a5=0.0080782d0
        a6=-0.0351117d0
        p=r-re
        vr=De-De*(1d0+a1*p+a2*p**2+a3*p**3+a4*p**4+
     &  a5*p**5+a6*p**6)*dexp(-a1*p)
        return
        end function

        subroutine arhflr(r1,rr,th,vlr)
        implicit real*8(a-h,o-z)
        integer i,n,l
        real*8,intent(in) :: r1,rr,th
        real*8,intent(out):: vlr
        real*8 p(0:8),c(6:12,0:8)
        parameter (alphaAr=11.08d0,nr=58)
        common/longrange/cdisp(6:12,0:2),ceq(6:12,0:8),rhf(nr),
     &  alpha1(nr),alpha2(nr),dip(nr),y1(nr),y2(nr),y3(nr)

        call splint(rhf,alpha1,y1,nr,r1,yalpha1)
        call splint(rhf,alpha2,y2,nr,r1,yalpha2)
        call splint(rhf,dip,y3,nr,r1,ydip)

        pi=dacos(-1d0)
        P(0)=1d0
        P(1)=dcos(th*pi/180d0)
        do i=1,7
        P(i+1)=((2*i+1)*dcos(th*pi/180d0)*P(i)-i*P(i-1))/(i+1)
        end do

        cind=ydip**2*alphaAr
        c=0d0
        c(6,0)=cdisp(6,0)*yalpha1/alpha1(14)+cind
        c(6,2)=cdisp(6,0)*yalpha2/yalpha1/3d0+cind
        do n=7,12
        do l=0,8
        c(n,l)=ceq(n,l)*cdisp(6,0)*yalpha1/alpha1(14)/
     &  (cdisp(6,0)+cind)
        end do;end do

        s2=0d0
        do n=6,12
        s1=0d0
        do l=0,8
          s1=s1+c(n,l)*p(l)
        end do
        s2=s2+s1/rr**n
        end do
        vlr=-s2

        end subroutine

        subroutine lrpes_init()
        implicit real*8(a-z)
        integer,parameter::nr=58
        data dy1,dyn/1d30,1d30/
        common/longrange/cdisp(6:12,0:2),ceq(6:12,0:8),rhf(nr),
     &  alpha1(nr),alpha2(nr),dip(nr),y1(nr),y2(nr),y3(nr)

        open(100,file='rhf.dat')
        open(101,file='alpha1.dat')
        open(102,file='alpha2.dat')
        open(103,file='dipole.dat')
        open(104,file='data_coeff_d.dat')
        do i=1,nr
         read(100,*)rhf(i)
         read(101,*)alpha1(i)
         read(102,*)alpha2(i)
         read(103,*)dip(i)
        enddo
        close(100);close(101);close(102);close(103)

!--- (2*alpha_orth+alpha_para)/3 polarizability part
        call spline(rhf,alpha1,nr,dy1,dyn,y1)

!--- (alpha_para-alpha_orth) polarizability part
        call spline(rhf,alpha2,nr,dy1,dyn,y2)

!--- Dipole moment part
        call spline(rhf,dip,nr,dy1,dyn,y3)

        do l=1,8
        read(104,*) (ceq(n,l),n=6,12)
        enddo
        cdisp=0d0
        cdisp(6,0)=36.75744d0
        cdisp(6,2)=2.783707d0
        close(104)
        endsubroutine


C##################################################################
C# SPLINE ROUTINES
C#            Numerical recipes in fortran
C#            Cambrige University Press
C#            York, 2nd edition, 1992.
C##################################################################
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      implicit double precision (a-h,o-z)
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
        if (h.eq.0.0d0) write(*,*) 'bad xa input in splint'
        a=(xa(khi)-x)/h
        b=(x-xa(klo))/h
        y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     &  2)/6.0d0
        return
        END
C#######################################################################
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
