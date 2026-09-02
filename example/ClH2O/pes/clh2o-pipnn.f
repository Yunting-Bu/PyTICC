!-->  program to get potential energy for a given geometry after NN fitting
!-->  global variables are declared in this module
       module nnparam
       implicit none
       real*8,parameter::alpha=1.0d0,vpesmin=-536.01052952d0,  !-->reactant
     $ vpescut=5.0d0,  ! 4eV
     % PI=3.141592653589793238d0,radian=PI/180.0d0,bohr=0.5291772d0
       integer,parameter::n0ho=20,n0hcl=18  !dipole
       integer,parameter::nxho=18,nxhcl=18  ! product energy
       integer,parameter::n2n=100
       real*8 DP0ho(n0ho),DP0hcl(n0hcl),R0ho(n0ho),R0hcl(n0hcl)
       real*8 Exhohcl(nxhcl,nxho),Rxho(nxho),Rxhcl(nxhcl)
       real*8 DP0ho2(n2n),DP0hcl2(n2n)
       real*8 Exhohcl2(nxhcl,nxho)
       integer,parameter::nbasis=18
       integer ninput,noutput,nhid,nlayer,ifunc,nwe,nodemax
       integer nterm(1:nbasis),nindex(1:nbasis,1:100,1:6)
       integer nscale
       integer, allocatable::nodes(:)

       real*8, allocatable::weighta(:,:,:),biasa(:,:)
       real*8, allocatable::pdela(:),pavga(:)
       real*8, allocatable::weightb(:,:,:),biasb(:,:)
       real*8, allocatable::pdelb(:),pavgb(:)
       real*8, allocatable::weightc(:,:,:),biasc(:,:)
       real*8, allocatable::pdelc(:),pavgc(:)

       end module nnparam

       subroutine clh2oNN(ct,vpes,vpesa,vpesb,vpesc) !ct in HHClO order and angstrom
       use nnparam
       implicit none
       integer,parameter::ndim=6
       integer j,k
       real*8 rb(ndim),xbond(ndim),basis(0:nbasis-1),tmp1
       real*8 txinput(1:nbasis-1)
       real*8 ct(3,4),xvec(3,ndim),xct(3,4),vpes
       real*8 vpesa,vpesb,vpesc,r12,r13,r14,r23,r24,r34

       basis=0.d0
       xct=ct
       
       xvec(:,1)=xct(:,2)-xct(:,1) !  H2->H1
       xvec(:,2)=xct(:,3)-xct(:,1) !  x->H1
       xvec(:,3)=xct(:,4)-xct(:,1) !  O->H1
       xvec(:,4)=xct(:,3)-xct(:,2) !  H2->x
       xvec(:,5)=xct(:,4)-xct(:,2) !  H2->O
       xvec(:,6)=xct(:,4)-xct(:,3) !  x-O 
       rb(1)=dsqrt(dot_product(xvec(:,1),xvec(:,1)))
       rb(2)=dsqrt(dot_product(xvec(:,2),xvec(:,2)))
       rb(3)=dsqrt(dot_product(xvec(:,3),xvec(:,3)))
       rb(4)=dsqrt(dot_product(xvec(:,4),xvec(:,4)))
       rb(5)=dsqrt(dot_product(xvec(:,5),xvec(:,5)))
       rb(6)=dsqrt(dot_product(xvec(:,6),xvec(:,6))) 

       r12=rb(1);r13=rb(2);r14=rb(3);r23=rb(4);r24=rb(5);r34=rb(6)

       xbond(:)=dexp(-rb(:)/alpha)
       
       call bemsav(xbond,basis)
      
       do j=1,nbasis-1
        txinput(j)=basis(j)
       enddo

       call getpota(txinput,vpesa)
       call getpotb(txinput,vpesb)
       call getpotc(txinput,vpesc)
       vpes=(vpesa+vpesb+vpesc)/3.0d0

       if(vpes.lt.-1.5d0)vpes=5.0d0
       vpes=min(vpes,5.0d0)
       
       return
        
       end subroutine clh2oNN

!-->  read NN weights and biases from matlab output
!-->  weights saved in 'weights.txt'
!-->  biases saved in 'biases.txt'
!-->  one has to call this subroutine one and only one before calling the getpot() subroutine
      subroutine pes_init
      use nnparam
      implicit none
      real*8,parameter::dy1=1.0d30,dyn=1.0d30
      integer,parameter::nn=100
      integer ihid,iwe,inode1,inode2,ilay1,ilay2
      integer ibasis,npd,iterm,ib,nfile
      integer i,j,k,m,n
      real*8 Et1,r1t,r2t,r3t,Et2,xa,xb,xc,vpes
      real*8 x1a(nxhcl),x2a(nxho),ya(nxhcl,nxho)
      real*8 ytmp(nn),y2tmp(nn),y2a(nxhcl,nxho)
      real*8 E1ho(nxho),E2hcl(nxhcl)
      character*100 fho,fhcl,line,f1
      
! HO and HCl separated at 50.0 angstrom      
      nfile=7
      open(222,status='scratch')
      open(nfile,file='weights.txt',status='old')
      rewind(nfile)
      read(nfile,'(a100)')line
      read(nfile,'(a100)')line
      do i=1,nxho 
        do j=1,nxhcl
         read(nfile,*)r1t,r2t,r3t,Et1,Et2
         Rxhcl(j)=r1t
         Rxho(i)=r2t
         Exhohcl(j,i)=Et1       

         write(222,'(a,i4,a,f16.8,a)')'Rxhcl(',j,')=',r1t,'d0'
         write(222,'(a,i4,a,f16.8,a)')'Rxho(',i,')=',r2t,'d0'
         write(222,'(a,i4,a,i4,a,f16.8,a)')'Exhohcl(',j,',',i,')=',
     % Et1,'d0'
         
        enddo
      enddo

      do i=1,nxho
       do j=1,nxhcl
        if(dabs(Rxhcl(j)-1.27413d0).le.1.0d-5)then
          E1ho(i)=Exhohcl(j,i)
         write(222,'(a,i4,a,f16.8,a)')'E1ho(',i,')=',Exhohcl(j,i),'d0'
        endif
        if(dabs(Rxho(i)-0.97131d0).le.1.0d-5)then
          E2hcl(j)=Exhohcl(j,i)
         write(222,'(a,i4,a,f16.8,a)')'E2hcl(',j,')=',Exhohcl(j,i),'d0'
        endif
       enddo
      enddo
      
      m=nxhcl
      n=nxho
      x1a=Rxhcl
      x2a=Rxho
      ya=Exhohcl
    
      do j=1,m 
       do k=1,n
        ytmp(k)= ya(j,k)
       enddo
       call spline(x2a,ytmp,n,1.0d30,1.0d30,y2tmp)
       do k=1,n
        y2a(j,k)=y2tmp(k)
       enddo
      enddo

      Exhohcl2=y2a
      
!HCl dipole      
      
      read(nfile,'(a100)')line
      read(nfile,'(a100)')line
      do i=1,n0hcl
       read(nfile,*)R0hcl(i),xa,xb,xc
       DP0hcl(i)=dsqrt(xa**2+xb**2+xc**2)     ! dipole in au
       write(222,'(a,i4,a,f16.8,a)')'DP0hcl(',i,')=',
     % dsqrt(xa**2+xb**2+xc**2),'d0'
       R0hcl(i)=R0hcl(i)/0.5291772d0            ! distance in au
       write(222,'(a,i4,a,f16.8,a)')'R0hcl(',i,')=',
     % R0hcl(i)/0.5291772d0,'d0'
      enddo

      call spline(R0hcl,DP0hcl,n0hcl,dy1,dyn,DP0hcl2)

!HO dipole      
     
      read(nfile,'(a100)')line
      read(nfile,'(a100)')line
      do i=1,n0ho
       read(nfile,*)R0ho(i),xa,xb,xc
       DP0ho(i)=dsqrt(xa**2+xb**2+xc**2)     ! dipole in au
       write(222,'(a,i4,a,f16.8,a)')'DP0ho(',i,')=',
     % dsqrt(xa**2+xb**2+xc**2),'d0'
       R0ho(i)=R0ho(i)/0.5291772d0            ! distance in au
       write(222,'(a,i4,a,f16.8,a)')'R0ho(',i,')=',
     % R0ho(i)/0.5291772d0,'d0'
      enddo

      call spline(R0ho,DP0ho,n0ho,dy1,dyn,DP0ho2)

!******************************************************************************
        open(4,file='biases.txt',status='old')
        rewind(4)
        read(4,'(a100)')line

!       open(nfile,file='weights.txt')

        read(nfile,'(a80)')line
      write(*,*)line
        read(nfile,*)ninput,nhid,noutput
        write(222,'(3(a,i3))')'ninput=',ninput,';nhid=',nhid,
     $ ';noutput=',noutput
        nscale=ninput+noutput
        nlayer=nhid+2 !additional one for input layer and one for output 
        write(222,'(a)')'nscale=ninput+noutput'
        write(222,'(a)')'nlayer=nhid+2'
        write(222,'(a)')'allocate(nodes(nlayer)'//
     %',pdela(nscale),pavga(nscale))'
        allocate(nodes(nlayer),pdela(nscale),pavga(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(nfile,*)(nodes(ihid),ihid=2,nhid+1)
        write(222,'(a)')'nodes(1)=ninput'
        write(222,'(a)')'nodes(nlayer)=noutput'
        do ihid=2,nhid+1
         write(222,'(a,i2,a,i2)')'nodes(',ihid,')=',nodes(ihid)
        enddo
        nodemax=0
        do i=1,nlayer
         nodemax=max(nodemax,nodes(i))
        enddo
        write(222,'(a)')'nodemax=0'
        write(222,'(a)')'do i=1,nlayer'
        write(222,'(a)')' nodemax=max(nodemax,nodes(i))'
        write(222,'(a)')'enddo'
        allocate(weighta(nodemax,nodemax,2:nlayer),
     %   biasa(nodemax,2:nlayer))
        read(nfile,*)ifunc,nwe
       write(222,'(a)')'allocate(weighta(nodemax,nodemax,2:nlayer),'//
     %'biasa(nodemax,2:nlayer))'
       write(222,'(a,i3,a,i6)')'ifunc=',ifunc,';nwe=',nwe
!-->....ifunc hence controls the type of transfer function used for hidden layers
!-->....At this time, only an equivalent transfer function can be used for all hidden layers
!-->....and the pure linear function is always applid to the output layer.
!-->....see function tranfun() for details
        read(nfile,*)(pdela(i),i=1,nscale)
        read(nfile,*)(pavga(i),i=1,nscale)
        do i=1,nscale
        write(222,'(a,i3,a,f20.16,a)')'pdela(',i,')=',pdela(i),'d0'
        write(222,'(a,i3,a,f20.16,a)')'pavga(',i,')=',pavga(i),'d0'
        enddo
        
        iwe=0
        write(222,'(a,i3)')'iwe=',iwe
        do ilay1=2,nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        do inode2=1,nodes(ilay2) !
        read(nfile,*)weighta(inode2,inode1,ilay1)
        write(222,'(a,i3,a,i3,a,i3,a, f40.20,a)')'weighta(',inode2,',',
     %  inode1,',',ilay1,')=',weighta(inode2,inode1,ilay1),'d0'
        iwe=iwe+1
        write(222,'(a)')'iwe=iwe+1'
        enddo
        read(4,*)biasa(inode1,ilay1)
        iwe=iwe+1
        write(222,'(a,i3,a,i3,a,f40.20,a)')'biasa(',inode1,',',ilay1,
     %')=',biasa(inode1,ilay1),'d0'
        write(222,'(a)')'iwe=iwe+1'
        enddo
        enddo
       
        write(222,'(a,i5)')'iwe=',iwe
        write(222,'(a)')'if (iwe.ne.nwe) then'
        write(222,'(a)')"write(*,*)'provided number of parameters ',nwe"
        write(222,'(a)')"write(*,*)'actual number of parameters ',iwe"
        write(222,'(a)')"write(*,*)'nwe not equal to iwe, check input"//
     ^  "files or code'"
        write(222,'(a)')"stop"
        write(222,'(a)')"endif"
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif

        read(4,'(a80)')line
        write(*,*)line
        read(nfile,'(a80)')line
        write(*,*)line
        read(nfile,*)ninput,nhid,noutput
        write(222,'(3(a,i3))')'ninput=',ninput,';nhid=',nhid,
     $ ';noutput=',noutput
        nscale=ninput+noutput
        nlayer=nhid+2 !additional one for input layer and one for output 
        write(222,'(a)')'nscale=ninput+noutput'
        write(222,'(a)')'nlayer=nhid+2'
        write(222,'(a)')'allocate(nodes(nlayer)'//
     %',pdelb(nscale),pavgb(nscale))'
        allocate(pdelb(nscale),pavgb(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(nfile,*)(nodes(ihid),ihid=2,nhid+1)
        write(222,'(a)')'nodes(1)=ninput'
        write(222,'(a)')'nodes(nlayer)=noutput'
        do ihid=2,nhid+1
         write(222,'(a,i2,a,i2)')'nodes(',ihid,')=',nodes(ihid)
        enddo
        nodemax=0
        do i=1,nlayer
         nodemax=max(nodemax,nodes(i))
        enddo
        write(222,'(a)')'nodemax=0'
        write(222,'(a)')'do i=1,nlayer'
        write(222,'(a)')' nodemax=max(nodemax,nodes(i))'
        write(222,'(a)')'enddo'
        allocate(weightb(nodemax,nodemax,2:nlayer),
     %   biasb(nodemax,2:nlayer))
        read(nfile,*)ifunc,nwe
       write(222,'(a)')'allocate(weightb(nodemax,nodemax,2:nlayer),'//
     %'biasb(nodemax,2:nlayer))'
       write(222,'(a,i3,a,i6)')'ifunc=',ifunc,';nwe=',nwe
!-->....ifunc hence controls the type of transfer function used for hidden layers
!-->....At this time, only an equivalent transfer function can be used for all hidden layers
!-->....and the pure linear function is always applid to the output layer.
!-->....see function tranfun() for details
        read(nfile,*)(pdelb(i),i=1,nscale)
        read(nfile,*)(pavgb(i),i=1,nscale)
        do i=1,nscale
        write(222,'(a,i3,a,f20.16,a)')'pdelb(',i,')=',pdelb(i),'d0'
        write(222,'(a,i3,a,f20.16,a)')'pavgb(',i,')=',pavgb(i),'d0'
        enddo
        iwe=0
        write(222,'(a,i3)')'iwe=',iwe
        do ilay1=2,nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        do inode2=1,nodes(ilay2) !
        read(nfile,*)weightb(inode2,inode1,ilay1)
        write(222,'(a,i3,a,i3,a,i3,a, f40.20,a)')'weightb(',inode2,',',
     %  inode1,',',ilay1,')=',weightb(inode2,inode1,ilay1),'d0'
        iwe=iwe+1
        write(222,'(a)')'iwe=iwe+1'
        enddo
        read(4,*)biasb(inode1,ilay1)
        iwe=iwe+1
        write(222,'(a,i3,a,i3,a,f40.20,a)')'biasb(',inode1,',',ilay1,
     %')=',biasb(inode1,ilay1),'d0'
        write(222,'(a)')'iwe=iwe+1'
        enddo
        enddo
        
        write(222,'(a,i5)')'iwe=',iwe
        write(222,'(a)')'if (iwe.ne.nwe) then'
        write(222,'(a)')"write(*,*)'provided number of parameters ',nwe"
        write(222,'(a)')"write(*,*)'actual number of parameters ',iwe"
        write(222,'(a)')"write(*,*)'nwe not equal to iwe, check input"//
     ^  "files or code'"
        write(222,'(a)')"stop"
        write(222,'(a)')"endif"
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif


        read(4,'(a80)')line
      write(*,*)line
        read(nfile,'(a80)')line
      write(*,*)line
        read(nfile,*)ninput,nhid,noutput
        write(222,'(3(a,i3))')'ninput=',ninput,';nhid=',nhid,
     $ ';noutput=',noutput
        nscale=ninput+noutput
        nlayer=nhid+2 !additional one for input layer and one for output 
        write(222,'(a)')'nscale=ninput+noutput'
        write(222,'(a)')'nlayer=nhid+2'
        write(222,'(a)')'allocate(nodes(nlayer)'//
     %',pdelc(nscale),pavgc(nscale))'
        allocate(pdelc(nscale),pavgc(nscale))
        nodes(1)=ninput
        nodes(nlayer)=noutput
        read(nfile,*)(nodes(ihid),ihid=2,nhid+1)
        write(222,'(a)')'nodes(1)=ninput'
        write(222,'(a)')'nodes(nlayer)=noutput'
        do ihid=2,nhid+1
         write(222,'(a,i2,a,i2)')'nodes(',ihid,')=',nodes(ihid)
        enddo
        nodemax=0
        do i=1,nlayer
         nodemax=max(nodemax,nodes(i))
        enddo
        write(222,'(a)')'nodemax=0'
        write(222,'(a)')'do i=1,nlayer'
        write(222,'(a)')' nodemax=max(nodemax,nodes(i))'
        write(222,'(a)')'enddo'
        allocate(weightc(nodemax,nodemax,2:nlayer),
     % biasc(nodemax,2:nlayer))
        read(nfile,*)ifunc,nwe
       write(222,'(a)')'allocate(weightc(nodemax,nodemax,2:nlayer),'//
     %'biasc(nodemax,2:nlayer))'
       write(222,'(a,i3,a,i6)')'ifunc=',ifunc,';nwe=',nwe
!-->....ifunc hence controls the type of transfer function used for hidden layers
!-->....At this time, only an equivalent transfer function can be used for all hidden layers
!-->....and the pure linear function is always applid to the output layer.
!-->....see function tranfun() for details
        read(nfile,*)(pdelc(i),i=1,nscale)
        read(nfile,*)(pavgc(i),i=1,nscale)
        do i=1,nscale
        write(222,'(a,i3,a,f20.16,a)')'pdelc(',i,')=',pdelc(i),'d0'
        write(222,'(a,i3,a,f20.16,a)')'pavgc(',i,')=',pavgc(i),'d0'
        enddo
        iwe=0
        write(222,'(a,i3)')'iwe=',iwe
        do ilay1=2,nlayer
        ilay2=ilay1-1
        do inode1=1,nodes(ilay1)
        do inode2=1,nodes(ilay2) !
        read(nfile,*)weightc(inode2,inode1,ilay1)
        write(222,'(a,i3,a,i3,a,i3,a, f40.20,a)')'weightc(',inode2,',',
     %  inode1,',',ilay1,')=',weightc(inode2,inode1,ilay1),'d0'
        iwe=iwe+1
        write(222,'(a)')'iwe=iwe+1'
        enddo
        read(4,*)biasc(inode1,ilay1)
        iwe=iwe+1
        write(222,'(a,i3,a,i3,a,f40.20,a)')'biasc(',inode1,',',ilay1,
     %')=',biasc(inode1,ilay1),'d0'
        write(222,'(a)')'iwe=iwe+1'
        enddo
        enddo
        
        write(222,'(a,i5)')'iwe=',iwe
        write(222,'(a)')'if (iwe.ne.nwe) then'
        write(222,'(a)')"write(*,*)'provided number of parameters ',nwe"
        write(222,'(a)')"write(*,*)'actual number of parameters ',iwe"
        write(222,'(a)')"write(*,*)'nwe not equal to iwe, check input"//
     ^  "files or code'"
        write(222,'(a)')"stop"
        write(222,'(a)')"endif"
        if (iwe.ne.nwe) then
           write(*,*)'provided number of parameters ',nwe
           write(*,*)'actual number of parameters ',iwe
           write(*,*)'nwe not equal to iwe, check input files or code'
           stop
        endif
        
        close(nfile)
        close(4)

        return

        end subroutine pes_init

        subroutine getpota(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun
!-->....set up the normalized input layer
c       write(*,*)ninput
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
     &*weighta(inode2,inode1,ilay1)
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
     &*weighta(inode2,inode1,ilay1)
        enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdela(nscale)+pavga(nscale)
        return
        end subroutine getpota

        subroutine getpotb(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun
!-->....set up the normalized input layer
c       write(*,*)ninput
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
     &*weightb(inode2,inode1,ilay1)
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
     &*weightb(inode2,inode1,ilay1)
        enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdelb(nscale)+pavgb(nscale)
        return
        end subroutine getpotb

        subroutine getpotc(x,vpot)
        use nnparam
        implicit none
        integer i,inode1,inode2,ilay1,ilay2
        real*8 x(ninput),y(nodemax,nlayer),vpot
        real*8, external :: tranfun
!-->....set up the normalized input layer
c       write(*,*)ninput
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
     &*weightc(inode2,inode1,ilay1)
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
     &*weightc(inode2,inode1,ilay1)
        enddo
!-->....the transfer function is linear y=x for output layer
!-->....so no operation is needed here
        enddo

!-->....the value of output layer is the fitted potntial 
        vpot=y(nodes(nlayer),nlayer)*pdelc(nscale)+pavgc(nscale)
        return
        end subroutine getpotc

        function tranfun(x,ifunc)
        implicit none
        integer ifunc
        real*8 tranfun,x
c    ifunc=1, transfer function is hyperbolic tangent function, 'tansig'
c    ifunc=2, transfer function is log sigmoid function, 'logsig'
c    ifunc=3, transfer function is pure linear function, 'purelin'. It is imposed to the output layer by default
        if (ifunc.eq.1) then
        tranfun=dtanh(x)
        else if (ifunc.eq.2) then
        tranfun=1d0/(1d0+exp(-x))
        else if (ifunc.eq.3) then
        tranfun=x
        endif
        return
        end

      function emsav(x,c) result(v)
      implicit none
      real*8,dimension(1:6)::x
      real*8,dimension(0:17)::c
      real*8::v
      ! ::::::::::::::::::::
      real*8,dimension(0:17)::p
      call bemsav(x,p)
      v = dot_product(p,c)
      return
      end function emsav
 
      subroutine bemsav(x,p)
      implicit none
      real*8,dimension(1:6),intent(in)::x
      real*8,dimension(0:17),intent(out)::p
      ! ::::::::::::::::::::
      real*8,dimension(0:10)::m
      call evmono(x,m)
      call evpoly(m,p)
      return
      end subroutine bemsav
 
      subroutine evmono(x,m)
      implicit none
      real*8,dimension(1:6),intent(in)::x
      real*8,dimension(0:10),intent(out)::m
 
      m(0)=1.D0
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
 
      subroutine evpoly(m,p)
      implicit none
      real*8,dimension(0:10),intent(in)::m
      real*8,dimension(0:17),intent(out)::p
 
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

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine HOHCl_pes_interface(xc,vpes)
      use nnparam,only: vpescut
      implicit none
      integer lxc,i,j
      real*8 xc(3,4),rbond(6),xct(3,4),xvec(3,6),xct2(3,4)
      real*8 dpthcl,dptho,r13,r23,rhpo,rhcl,vfit,vdp
      real*8 vt1,vcorr,vpes,tmp1,ybond(6),va,vb,vc

      real*8 xcom1(3),xcom2(3),rcom,mass(4),skey

      mass(1)=1.00794d0
      mass(2)=mass(1)
      mass(3)=35.453d0   !HHClO standard weight
      mass(4)=15.9994d0
      
      vt1=0.0d0
      vcorr=0.0d0
      xct=xc

      xvec(:,2)=xct(:,3)-xct(:,1) !  Cl-H1
      xvec(:,4)=xct(:,3)-xct(:,2) !  H2-Cl
      r13=dsqrt(dot_product(xvec(:,2),xvec(:,2)))
      r23=dsqrt(dot_product(xvec(:,4),xvec(:,4)))

! check the linked atom order: make sure that
! 1st H linked or near Cl 

      if(r13.gt.r23)then
       xct2(:,1)=xct(:,2) 
       xct2(:,2)=xct(:,1) 
       xct2(:,3)=xct(:,3) 
       xct2(:,4)=xct(:,4)
      else
       xct2=xct
      endif

      call before_vpes(xct2,lxc)  !xct in order for H1-Cl  and H2-O, if possible

      if(lxc.eq.-1)then
        vpes=vpescut
        return
      endif

!     lxc=0
      if(lxc.eq.2)then
        call HO_HCl_correction(xct2,vpes)        
        vpes=min(vpes,vpescut)
        return
      endif

      if(lxc.eq.0)then
        call clh2oNN(xct2,vpes,va,vb,vc)
        vpes=min(vpes,vpescut)
        return
      endif

      if(lxc.eq.1)then
! check the linked atom order: make sure that
! 1st H linked or near Cl 

        xcom1(:)=xct2(:,1)+(mass(3)/(mass(1)+mass(3)))*
     $ (xct2(:,3)-xct2(:,1))
        xcom2(:)=xct2(:,2)+(mass(4)/(mass(2)+mass(4)))*
     $ (xct2(:,4)-xct2(:,2))
        rcom=dsqrt(dot_product(xcom1-xcom2,xcom1-xcom2))
        skey=(1.0d0-dtanh(0.8d0*(rcom-10.0d0)))/2.0d0 
        call clh2oNN(xct2,vfit,va,vb,vc)
        call HO_HCl_correction(xct2,vdp)
        vpes=skey*vfit+(1.0d0-skey)*vdp 
        vpes=min(vpes,vpescut)
        return
      endif

      return
      end subroutine HOHCl_pes_interface 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!make sure that 1st H linked or near to Cl
      subroutine HO_HCl_correction(xc,vdp)
      use nnparam
      implicit none 
      real*8 xc(3,4),xct2(3,4),vdp,vcorr,rbond(6),vt1
      real*8 xvec(3,6),rhpo,rhcl,dpthcl,dptho
      real*8 angB,angA,angAB,vecHCl(3),vecHO(3),vecHH(3)
      real*8 pointA(3),pointB(3),vecACl(3),vecBO(3),vecAB(3)
      real*8 rACl,rBO,rAB

      xct2=xc
      vt1=0.0d0
      vcorr=0.d0
      
      xvec(:,1)=xct2(:,2)-xct2(:,1) !  H2->H1
      xvec(:,2)=xct2(:,3)-xct2(:,1) !  Cl->H1
      xvec(:,3)=xct2(:,4)-xct2(:,1) !  O->H1
      xvec(:,4)=xct2(:,3)-xct2(:,2) !  H2->Cl
      xvec(:,5)=xct2(:,4)-xct2(:,2) !  H2->O
      xvec(:,6)=xct2(:,4)-xct2(:,3) !  Cl-O 
     
      rbond(1)=dsqrt(dot_product(xvec(:,1),xvec(:,1)))
      rbond(2)=dsqrt(dot_product(xvec(:,2),xvec(:,2)))
      rbond(3)=dsqrt(dot_product(xvec(:,3),xvec(:,3)))
      rbond(4)=dsqrt(dot_product(xvec(:,4),xvec(:,4)))
      rbond(5)=dsqrt(dot_product(xvec(:,5),xvec(:,5)))
      rbond(6)=dsqrt(dot_product(xvec(:,6),xvec(:,6))) 

      rhcl=rbond(2)
      rhpo=rbond(5)

!     Cl /\                       /\ O     
!          \                     /
!           \ thetaA            /thetaB
!         A  \-----------------/-----/
!             \               / B
!              \             /
!               \           / H
!             H  \

             
      call HO_HCl_pes_spline(rhcl,rhpo,vt1) 

      rhcl=rhcl/0.5291772d0
      rhpo=rhpo/0.52917720d0

      call hcl_dipole(rhcl,dpthcl)
      call ho_dipole(rhpo,dptho)

!      dipole should be for the middle point of HCl and HO 

      vecHCl(:)=xvec(:,2)
      vecHO(:)=xvec(:,5)
         
      pointA(:)=xct2(:,1)+vecHCl(:)/2.0d0
      pointB(:)=xct2(:,2)+vecHO(:)/2.0d0

      vecACl(:)=xct2(:,3)-pointA(:)
      vecBO(:)=xct2(:,4)-pointB(:)
      vecAB(:)=pointB(:)-pointA(:)

      rACl=dsqrt(dot_product(vecACl(:),vecACl(:)))
      rBO=dsqrt(dot_product(vecBO(:),vecBO(:)))
      rAB=dsqrt(dot_product(vecAB(:),vecAB(:)))

      angA=dot_product(vecACl,vecAB)/(rACl*rAB)
      angA=dacos(max(-1.0d0,min(1.0d0,angA)))

      angB=dot_product(vecBO,vecAB)/(rBO*rAB)
      angB=dacos(max(-1.0d0,min(1.0d0,angB)))

      angAB=dot_product(vecACl,vecBO)/(rACl*rBO)
      angAB=dacos(max(-1.0d0,min(1.0d0,angAB)))

      vcorr=dpthcl*dptho/((rAB/0.5291772d0)**3)*
     $      (dcos(angAB)-3.0d0*dcos(angB)*dcos(angA))

      vdp=((vt1+vcorr)-vpesmin)*627.509d0 - 0.15d0
      vdp=vdp/23.0605d0

      return

      end subroutine HO_HCl_correction
      
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! determine if need to call correction from dipole interaction of HO and HCl
! or, switch HHFO in the order of H1-Cl3, H2-O4 if possible
      subroutine before_vpes(xct,lxc)
      implicit none
      real*8 xct(3,4),bt(6),xvec(3,6),xct2(3,4)
      real*8 xcom1(3),xcom2(3),rcom,mass(4)
      real*8 r12,r13,r14,r23,r24,r34
      integer lxc,i,lbc,lcd,lde

      mass(1)=1.00794d0
      mass(2)=mass(1)
      mass(3)=35.453d0   !HHClO standard weight
      mass(4)=15.9994d0
      
! lxc=-1, unphysical point
! lxc=0,  normal
! lxc=1,  HO+HCl, using switch function
! lxc=2,  HO+HCl, using dipole interaction correction

      lxc=0
      lbc=0
      lcd=0
      lde=0
     
      xct2=xct
      
      xvec(:,1)=xct2(:,2)-xct2(:,1) !  H2-H1
      xvec(:,2)=xct2(:,3)-xct2(:,1) !  Cl-H1
      xvec(:,3)=xct2(:,4)-xct2(:,1) !  O-H1
      xvec(:,4)=xct2(:,3)-xct2(:,2) !  H2-Cl
      xvec(:,5)=xct2(:,4)-xct2(:,2) !  H2-O
      xvec(:,6)=xct2(:,4)-xct2(:,3) !  Cl-O 
     
      r12=dsqrt(dot_product(xvec(:,1),xvec(:,1)))
      r13=dsqrt(dot_product(xvec(:,2),xvec(:,2)))
      r14=dsqrt(dot_product(xvec(:,3),xvec(:,3)))
      r23=dsqrt(dot_product(xvec(:,4),xvec(:,4)))
      r24=dsqrt(dot_product(xvec(:,5),xvec(:,5)))
      r34=dsqrt(dot_product(xvec(:,6),xvec(:,6))) 

      bt(:)=(/r12,r13,r14,r23,r24,r34/)

! remove some unphysical points      
      
      do i=1,6
       if(bt(i).le.0.60d0)then
         lxc=-1 
         return
       endif
      enddo

!--> ClO distance, do not consider  H2 + ClO channel      
!--> ClO equilibrium distance,1.596gstrom, x 130% = 2.07 angstrom
      if(r34.le.1.30d0.or.r12.le.0.60d0) then 
       lxc=-1
       return
      endif

      if(minval(bt).ge.2.00d0)then  ! 
        lxc=-1
        return
      endif

      if(r13.le.0.75d0.or.r23.le.0.75d0)then
        lxc=-1
        return
      endif

! H+O+HCl or HO + H +Cl or H+H+Cl+O      
      if(r14.lt.r24.and.r24.gt.2.0d0) then
        if(r14.ge.2.0d0.or.r23.ge.2.0d0)then   
          lxc=-1
          return
        endif
      endif

      if(r24.lt.r14.and.r14.gt.2.0d0) then
        if(r24.ge.2.0d0.or.r13.ge.2.0d0)then  
          lxc=-1
          return
        endif
      endif
! OH1 and OH2 are both large
      if(r14.ge.2.0d0.and.r24.ge.2.0d0)then
        lxc=-1
        return
      endif
     
! ClH1 and ClH2 are both small
      if(r13.le.1.1d0.and.r23.le.1.1d0)then
        lxc=-1
        return
      endif
     
!     if(r23.lt.r13.and.r13.gt.1.9d0.and.r24.gt.1.8d0) then
!       if(r14.ge.1.8d0.or.r23.ge.1.9d0)then   
!         lxc=-1
!         return
!       endif
!     endif

!     if(r13.lt.r23.and.r23.gt.1.9d0.and.r14.gt.1.8d0) then
!       if(r13.ge.1.8d0.or.r24.ge.1.9d0)then  
!         lxc=-1
!         return
!       endif
!     endif
     
!check if HO and HCl were produced

!     if(r12.ge.3.5d0)then
!       
!       xcom1(:)=xct2(:,1)+(mass(3)/(mass(1)+mass(3)))*
!    $ (xct2(:,3)-xct2(:,1))
!       xcom2(:)=xct2(:,2)+(mass(4)/(mass(2)+mass(4)))*
!    $ (xct2(:,4)-xct2(:,2))
!    
!       rcom=dsqrt(dot_product(xcom1-xcom2,xcom1-xcom2))
!     
!       if(rcom.ge.5.0d0.and.rcom.le.16.0d0)then
!         lxc=1
!         return
!       else if(rcom.gt.16.0d0)then
!         lxc=2
!         return
!       else
!         lxc=0
!         return
!       endif

!     endif 
      lxc=0

      return

      end subroutine before_vpes
      
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine ho_dipole(rt,dpt)
      use nnparam
      IMPLICIT none
      integer,parameter::nn=100
      real*8,parameter::dy1=1.0d30,dyn=1.0d30
      real*8 y2(nn),rt,dpt

      y2=DP0ho2
      if (rt.lt.R0ho(1))  rt=R0ho(1)
      if (rt.gt.R0ho(n0ho))rt=R0ho(n0ho)
      call splint(R0ho,DP0ho,y2,n0ho,rt,dpt)

      return
        
      end  subroutine ho_dipole
     
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine hcl_dipole(rt,dpt)
      use nnparam
      IMPLICIT none
      integer,parameter::nn=100
      real*8,parameter::dy1=1.0d30,dyn=1.0d30
      real*8 y2(nn),rt,dpt

      y2=DP0hcl2
      if (rt.lt.R0hcl(1))  rt=R0hcl(1)
      if (rt.gt.R0hcl(n0hcl))rt=R0hcl(n0hcl)
      call splint(R0hcl,DP0hcl,y2,n0hcl,rt,dpt)

      return
        
      end  subroutine hcl_dipole
  
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine HO_HCl_pes_spline(r1,r2,vpes)
      use nnparam
      implicit none
      integer,parameter::nn=100 
      real*8 ytmp(nn),y2tmp(nn),y2a(nxhcl,nxho),yytmp(nn)
      real*8 x1a(nxhcl),x2a(nxho),ya(nxhcl,nxho)
      integer i,j,k,m,n
      real*8 r1,r2,x1,x2,y,vpes
      
      x1=r1
      x2=r2
      
      x1a=Rxhcl
      x2a=Rxho
      ya=Exhohcl

      m=nxhcl
      n=nxho

      y2a=Exhohcl2 
      
      do j=1,m
       do k=1,n
        ytmp(k)=ya(j,k)
        y2tmp(k)=y2a(j,k)
       enddo

       call splint(x2a,ytmp,y2tmp,n,x2,yytmp(j))

      enddo

      call spline(x1a,yytmp,m,1.0d30,1.0d30,y2tmp)
      call splint(x1a,yytmp,y2tmp,m,x1,y)

      vpes=y
      
      return

      end subroutine  HO_HCl_pes_spline
      
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cC##################################################################
c      
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
c 1    if (khi-klo.gt.1) then
c        k=(khi+klo)/2
c        if(xa(k).gt.x)then
c          khi=k
c        else
c          klo=k
c        endif
c      goto 1
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
cC##############################################################################
c      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
c      implicit double precision  (a-h,o-z)
c      DIMENSION x(n),y(n),y2(n)
c      PARAMETER (NMAX=100)
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
c
c!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->!-->
