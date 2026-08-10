!-->  program to get potential energy for a given geometry after NN fitting
!-->  global variables are declared in this module
!-->  2023-2-11 change by junyan read nine nn parameter in same time
        module nnparam
        implicit none
        integer ninput,noutput,nhid,nlayer,ifunc,nwe,nodemax
        integer nscale
        integer, allocatable::nodes(:)
        real*8, allocatable::weight(:,:,:),bias(:,:)
        real*8, allocatable::weight1(:,:,:),bias1(:,:)
        real*8, allocatable::weight2(:,:,:),bias2(:,:)
        real*8, allocatable::weight3(:,:,:),bias3(:,:)
        real*8, allocatable::weight4(:,:,:),bias4(:,:)
        real*8, allocatable::weight5(:,:,:),bias5(:,:)
        real*8, allocatable::weight6(:,:,:),bias6(:,:)
        real*8, allocatable::weight7(:,:,:),bias7(:,:)
        real*8, allocatable::weight8(:,:,:),bias8(:,:)
        real*8, allocatable::weight9(:,:,:),bias9(:,:)
        real*8, allocatable::pdel(:),pavg(:)
        real*8, allocatable::pdel1(:),pavg1(:)
        real*8, allocatable::pdel2(:),pavg2(:)
        real*8, allocatable::pdel3(:),pavg3(:)
        real*8, allocatable::pdel4(:),pavg4(:)
        real*8, allocatable::pdel5(:),pavg5(:)
        real*8, allocatable::pdel6(:),pavg6(:)
        real*8, allocatable::pdel7(:),pavg7(:)
        real*8, allocatable::pdel8(:),pavg8(:)
        real*8, allocatable::pdel9(:),pavg9(:)
        end module nnparam

!-->  read NN weights and biases from matlab output
!-->  weights saved in 'weights.txt'
!-->  biases saved in 'biases.txt'
!-->  one has to call this subroutine once and only once before calling the getpot() subroutine
        subroutine pes_init
        use nnparam
        implicit none
        integer i,ihid,iwe,inode1,inode2,ilay1,ilay2
        character f1*80
        open(111,file='weights1.txt')
        open(222,file='biases1.txt')
        open(333,file='weights2.txt')
        open(444,file='biases2.txt')
        open(555,file='weights3.txt')
        open(666,file='biases3.txt')
        open(777,file='weights4.txt')
        open(888,file='biases4.txt')
        open(999,file='weights5.txt')
        open(101010,file='biases5.txt')
        open(111111,file='weights6.txt')
        open(121212,file='biases6.txt')
        open(131313,file='weights7.txt')
        open(141414,file='biases7.txt')
        open(151515,file='weights8.txt')
        open(161616,file='biases8.txt')
        open(171717,file='weights9.txt')
        open(181818,file='biases9.txt')

        read(111,*)ninput,nhid,noutput
        read(333,*)
        read(555,*)
        read(777,*)
        read(999,*)
        read(111111,*)
        read(131313,*)
        read(151515,*)
        read(171717,*)
        
        nscale=ninput+noutput
        nlayer=nhid+2

        allocate(nodes(nlayer),pdel(nscale),pavg(nscale))
        allocate(pdel1(nscale),pavg1(nscale))
        allocate(pdel2(nscale),pavg2(nscale))
        allocate(pdel3(nscale),pavg3(nscale))
        allocate(pdel4(nscale),pavg4(nscale))
        allocate(pdel5(nscale),pavg5(nscale))
        allocate(pdel6(nscale),pavg6(nscale))
        allocate(pdel7(nscale),pavg7(nscale))
        allocate(pdel8(nscale),pavg8(nscale))
        allocate(pdel9(nscale),pavg9(nscale))
        
        nodes(1)=ninput
        nodes(nlayer)=noutput

        read(111,*)(nodes(ihid),ihid=2,nhid+1)
        read(333,*)
        read(555,*)
        read(777,*)
        read(999,*)
        read(111111,*)
        read(131313,*)
        read(151515,*)
        read(171717,*)

        nodemax=0
        do i=1,nlayer
        nodemax=max(nodemax,nodes(i))
        enddo

       allocate(weight(nodemax,nodemax,2:nlayer),bias(nodemax,2:nlayer))
       allocate(weight1(nodemax,nodemax,2:nlayer),
     &    bias1(nodemax,2:nlayer))
       allocate(weight2(nodemax,nodemax,2:nlayer),
     &    bias2(nodemax,2:nlayer))
       allocate(weight3(nodemax,nodemax,2:nlayer),
     &    bias3(nodemax,2:nlayer))
       allocate(weight4(nodemax,nodemax,2:nlayer),
     &    bias4(nodemax,2:nlayer))
       allocate(weight5(nodemax,nodemax,2:nlayer),
     &    bias5(nodemax,2:nlayer))
       allocate(weight6(nodemax,nodemax,2:nlayer),
     &    bias6(nodemax,2:nlayer))
       allocate(weight7(nodemax,nodemax,2:nlayer),
     &    bias7(nodemax,2:nlayer))
       allocate(weight8(nodemax,nodemax,2:nlayer),
     &    bias8(nodemax,2:nlayer))
       allocate(weight9(nodemax,nodemax,2:nlayer),
     &    bias9(nodemax,2:nlayer))



        read(111,*)ifunc,nwe
        read(333,*)
        read(555,*)
        read(777,*)
        read(999,*)
        read(111111,*)
        read(131313,*)
        read(151515,*)
        read(171717,*)

!-->....ifunc hence controls the type of transfer function used for hidden layers
!-->....At this time, only an equivalent transfer function can be used for all hidden layers
!-->....and the pure linear function is always applid to the output layer.
!-->....see function tranfun() for details

        read(111,*)(pdel1(i),i=1,nscale)
        read(111,*)(pavg1(i),i=1,nscale)
        read(333,*)(pdel2(i),i=1,nscale)
        read(333,*)(pavg2(i),i=1,nscale)
        read(555,*)(pdel3(i),i=1,nscale)
        read(555,*)(pavg3(i),i=1,nscale)
        read(777,*)(pdel4(i),i=1,nscale)
        read(777,*)(pavg4(i),i=1,nscale)
        read(999,*)(pdel5(i),i=1,nscale)
        read(999,*)(pavg5(i),i=1,nscale)
        read(111111,*)(pdel6(i),i=1,nscale)
        read(111111,*)(pavg6(i),i=1,nscale)
        read(131313,*)(pdel7(i),i=1,nscale)
        read(131313,*)(pavg7(i),i=1,nscale)
        read(151515,*)(pdel8(i),i=1,nscale)
        read(151515,*)(pavg8(i),i=1,nscale)
        read(171717,*)(pdel9(i),i=1,nscale)
        read(171717,*)(pavg9(i),i=1,nscale)
        
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
        read(111,*)weight1(inode2,inode1,ilay1)
        read(333,*)weight2(inode2,inode1,ilay1)
        read(555,*)weight3(inode2,inode1,ilay1)
        read(777,*)weight4(inode2,inode1,ilay1)
        read(999,*)weight5(inode2,inode1,ilay1)
        read(111111,*)weight6(inode2,inode1,ilay1)
        read(131313,*)weight7(inode2,inode1,ilay1)
        read(151515,*)weight8(inode2,inode1,ilay1)
        read(171717,*)weight9(inode2,inode1,ilay1)
        enddo
        read(222,*)bias1(inode1,ilay1)
        read(444,*)bias2(inode1,ilay1)
        read(666,*)bias3(inode1,ilay1)
        read(888,*)bias4(inode1,ilay1)
        read(101010,*)bias5(inode1,ilay1)
        read(121212,*)bias6(inode1,ilay1)
        read(141414,*)bias7(inode1,ilay1)
        read(161616,*)bias8(inode1,ilay1)
        read(181818,*)bias9(inode1,ilay1)
        
        enddo
        enddo
        write(*,*)'read all parameters done'
        close(111)
        close(222)
        close(333)
        close(444)
        close(555)
        close(666)
        close(777)
        close(888)
        close(999)
        close(101010)
        close(111111)
        close(121212)
        close(131313)
        close(141414)
        close(151515)
        close(161616)
        close(171717)
        close(181818)
        end subroutine pes_init

        subroutine getpot(x,vpot,n)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2,n
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun
!-->....set up the normalized input layer


        if (n == 1) then
           pavg = pavg1
           pdel = pdel1
           weight = weight1
           bias = bias1
        endif
         
        if (n == 2) then
           pavg = pavg2
           pdel = pdel2
           weight = weight2
           bias = bias2
        endif

        if (n == 3) then
           pavg = pavg3
           pdel = pdel3
           weight = weight3
           bias = bias3
        endif

        if (n == 4) then
           pavg = pavg4
           pdel = pdel4
           weight = weight4
           bias = bias4
        endif

        if (n == 5) then
           pavg = pavg5
           pdel = pdel5
           weight = weight5
           bias = bias5
        endif

        if (n == 6) then
           pavg = pavg6
           pdel = pdel6
           weight = weight6
           bias = bias6
        endif

        if (n == 7) then
           pavg = pavg7
           pdel = pdel7
           weight = weight7
           bias = bias7
        endif
        
        if (n == 8) then
           pavg = pavg8
           pdel = pdel8
           weight = weight8
           bias = bias8
        endif

        if (n == 9) then
           pavg = pavg9
           pdel = pdel9
           weight = weight9
           bias = bias9
        endif


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
     &*weight(inode2,inode1,ilay1)
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
     &*weight(inode2,inode1,ilay1)
        enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdel(nscale)+pavg(nscale)
        return
        end

        function tranfun(x,ifunc)
        implicit none
        integer ifunc
        real*8 tranfun,x
c    ifunc=1, transfer function is hyperbolic tangent function, 'tansig'
c    ifunc=2, transfer function is log sigmoid function, 'logsig'
c    ifunc=3, transfer function is pure linear function, 'purelin'. It is imposed to the output layer by default
        if (ifunc.eq.1) then
        tranfun=tanh(x)
        else if (ifunc.eq.2) then
        tranfun=1d0/(1d0+exp(-x))
        else if (ifunc.eq.3) then
        tranfun=x
        endif
        return
        end


      subroutine pot0(npes,vdia,x)
      implicit none
      integer :: npes
      real*8 :: vdia(npes,npes)
      real*8 :: x(3)
      real*8 :: R_jacobi,theta_jacobi,roo,roh1,roh2
      real*8 :: a1,a2,a3
      real*8 :: vlr,switch
      real*8 :: vx1,vx2,vx3
      real*8 :: v11,v22,v12,ad1,ad2
      real*8 :: dampv12
      real*8 :: v(9)
      real*8 :: DPME(2,2),eigen(2),work(20)
      real*8 :: pi = dacos(-1.d0)
      integer :: info,nn

      roh1 = x(1)       
      roh2 = x(3)
      roo = x(2)

      a1 = (roh2**2+roo**2-roh1**2)/(2.d0*roh2*roo)
      R_jacobi = dsqrt(roh2**2+ (roo/2.d0)**2 - roh2*roo*a1)
      a2 = (R_jacobi**2+(roo/2.d0)**2-roh1**2)/(R_jacobi*roo)

      if (a2 > 1.d0) a2 = 1.d0
      if (a2 < -1.d0) a2 = -1.d0     
      theta_jacobi = dacos(a2)*180.d0/pi

      if (R_jacobi == 0.d0) theta_jacobi = 90.d0

      do nn = 1,9
         call distopip(roh1,roh2,roo,v(nn),nn)
      enddo

      v11 = sum(v(1:3))/3.d0
      v22 = sum(v(4:6))/3.d0
      v12 = sum(v(7:9))*dsin(2.d0*theta_jacobi/180.d0*pi)/3.d0

      if (R_jacobi > 6.d0) then
        v12 = (1.d0-dampv12(R_jacobi))*v12 + dampv12(R_jacobi)*0.d0
      endif
      
      if (roo > 6.d0) then
        v12 = (1.d0-dampv12(roo))*v12 + dampv12(roo)*0.d0
      endif

      if (R_jacobi .ge. 7.d0 .or. roo .ge. 7.d0 ) v12 = 0.d0

      DPME(1,1) = v11-2.367d0+2.930d0
      DPME(1,2) = v12
      DPME(2,1) = v12
      DPME(2,2) = v22-2.367d0+2.930d0

      DPME = DPME/27.21138d0

      if (npes == 2) vdia = DPME
      if (npes == 1) then
         call dsyev('V','U',2,DPME,2,eigen,work,10*2,info)
         vdia(npes,npes) = eigen(1)
      endif

      return
      end subroutine
      

      subroutine Jacobi_to_bond(R,roo,theta,r1,r2,angle)
      implicit none
      real*8,intent(in)  :: R,roo,theta
      real*8 :: theta_rad
      real*8,intent(out) :: r1,r2,angle
      real*8,parameter   :: pi = dacos(-1.d0)

         theta_rad = theta*pi/180.d0
         r1 = (R**2 + 0.25d0*roo**2 - R*roo*dcos(theta_rad))**0.5
         r2 = (R**2 + 0.25d0*roo**2 + R*roo*dcos(theta_rad))**0.5
         angle = (r1**2+r2**2-roo**2)/(2.d0*r1*r2)

      return
      end subroutine

      subroutine distopip(rr1,rr2,rr3,vx,n)
      implicit none
      integer :: i,ios,total,j,k,ijk,n
      double precision :: r(3),v0,vx,dd,var(1,1),pip(3),r1,r2
      real*8 :: theta,a,b,c,rr1,rr2,rr3,ccth,th,rrr1,rrr2
      real*8 :: g(3),vx_ev
      double precision :: bohr=0.5291772108d0
      double precision :: alpha  = 2.d0
      double precision :: H2eV=27.21138d0

      r(1) = rr1
      r(2) = rr2
      r(3) = rr3
      r(1)=exp(-(r(1))/(alpha))
      r(2)=exp(-(r(2))/(alpha))
      r(3)=exp(-(r(3))/(alpha))

      pip(1)=(r(1)+r(2))/2
      pip(2)=r(1)*r(2)
      pip(3)=r(3)

      call getpot(pip,vx,n)

      return
      end subroutine
       
      subroutine potential_AB(r,v,istate)
      implicit none
      integer :: istate
      real*8 :: r,v,DPME(2,2),x(3)

      x(1) = 30000.d0
      x(2) = r
      x(3) = 30000.d0
      
      call pot0(2,DPME,x)

      v = DPME(istate,istate)

      return
      end subroutine

      subroutine interaction_potential_ABC(RR,roo,th,vout,switch)
      implicit none
      real*8 :: RR,roo,th,r1,r2,angle
      real*8 :: switchR,vlr
      real*8 :: x(3)
      real*8 :: dpme(2,2)
      real*8 :: v,vout
      integer :: i,switch
      
      call Jacobi_to_bond(RR,roo,th,r1,r2,angle)
      x(1) = r1
      x(2) = roo
      x(3) = r2

      call pot0(2,dpme,x)

      do i = 1,2
        call potential_AB(roo,v,i)
        dpme(i,i) = dpme(i,i) - v
      enddo

      call get_long(th,RR,roo,vlr)

      if (roo < 3.d0 .and. roo > 1.5d0) then
          dpme(1,1) = vlr/27.21138d0/8065.5d0*switchR(RR) 
     &  + (1.d0-switchR(RR))*dpme(1,1)
      endif 

      if (switch == 1) vout = dpme(1,1)
      if (switch == 2) vout = dpme(2,2)
      if (switch == 3) vout = dpme(1,2)

      return
      end


      function dampv12(R_jacobi)
      implicit none
      real*8 :: dampv12,R_jacobi

      dampv12 = 1.d0/(1.d0 + dexp(-10.0*(R_jacobi - 6.5d0)))

      if (R_jacobi <= 6.d0) dampv12 = 0.d0
      if (R_jacobi >= 7.d0) dampv12 = 1.d0

      return
      end function
 
      function switchR(R_jacobi)
      implicit none
      real*8 :: switchR,R_jacobi

      switchR = 1.d0/(1.d0+dexp(-2.0d0*(R_jacobi-11.d0)))

      if (R_jacobi >= 14.d0) switchR = 1.d0
      if (R_jacobi <= 8.d0) switchR = 0.d0

      return
      end function
