
c   ab inito potential energy surface for adiabatic electronic state of H2O
c      Please add the following statement in your main program:
c           call potread
c     use the following statement to calculate the PES when needed:
c                 call h2opes(r1,r2,cth,v,icase)
c      where r1 and r2 are the two bondlengthis of O-H in au,
c            th is the enclosed angle in degree, cth=cos(th)
c       v is the potential in hartree 

 
	subroutine potreaddia
c   in bondlength coordinates (theta,roh1,roh2)
        implicit real*8(a-h,o-z)
        parameter (pi=3.141592653589793d0)    
        parameter (nroh1=19,nth=19,nroh2=70,ip=4)
	parameter (m=100,n=100,l=100)
	dimension xt(m),y(m),y2(l)  
        data vmin1/-76.36385566d0/
       data dy1,dyn/1.0d30,1.0d30/
	common /bvcutd/vcut
        common /pesd/pesmin,roh2(nroh1,nth,nroh2),roh1(nroh1),thth(nth),
     &  ve(ip,nroh1,nth,nroh2),ind(nroh1,nth)
        open(98,file='new8-dia.dat')
        open(66,file='check-dia.txt')
        vcut=15d0/27.2114d0
        pesmin=10000
        do i=1,nroh1
          do j=1,nth
           ind(i,j)=0
            do k=1,nroh2
            read(98,*)mm,ra1,th1,ra2,ve1,ve2,ve3,ve4
           if (ra1+ra2.lt.0.1) goto 7
            if (k.eq.1) then
            rra1=ra1
            rra2=ra2
            tth1=th1
            else
            dal=abs(ra1-rra1)+abs(th1-tth1)
            if (dal.gt.0.01) then
            write(*,*)' error  ',ii,ra1,rra1,ra2,rra2
            stop
            endif
            endif
            if (ve1.lt.pesmin) then
            pesmin=ve1
            r1m=ra1
            r2m=ra2
            thm=th1
            endif
           ind(i,j)=ind(i,j)+1
            roh2(i,j,k)=ra2
            roh1(i)=ra1
            thth(j)=th1
            ve(1,i,j,k)=ve1-vmin1
            ve(2,i,j,k)=ve2-vmin1
	    ve(3,i,j,k)=ve3
            ve(4,i,j,k)=ve4-vmin1
            enddo
7          continue
          enddo  
        enddo
        write(55,*)r1m,r2m,thm
        write(*,*)r1m,r2m,thm,pesmin
        write(*,*)' th1=',(thth(i),i=1,nth)
        write(*,*)' roh1=',(roh1(i),i=1,nroh1)
          do i=1,nroh1
          do j=1,nth
           do k=1,ind(i,j)
          if (ve(1,i,j,k).gt.vcut) ve(1,i,j,k)=vcut
          if (ve(2,i,j,k).gt.vcut) ve(2,i,j,k)=vcut
          if (ve(4,i,j,k).gt.vcut) ve(4,i,j,k)=vcut
          write(66,1001)roh1(i),thth(j),roh2(i,j,k)
     &,ve(1,i,j,k),ve(2,i,j,k),ve(3,i,j,k),ve(4,i,j,k)
 1001      format (3f8.3,6f12.8)
           enddo
         enddo
       enddo
       close(98)
       close(66)
       end


	subroutine potreadadi
c   in bondlength coordinates (theta,roh1,roh2)
        implicit real*8(a-h,o-z)
        parameter (pi=3.141592653589793d0)    
        parameter (nroh1=19,nth=19,nroh2=70,ip=3)
	parameter (m=100,n=100,l=100)
	dimension xt(m),y(m),y2(l)  
        data vmin1/-76.36385566d0/
       data dy1,dyn/1.0d30,1.0d30/
	common /bvcut/vcut
        common /pes/pesmin,roh2(nroh1,nth,nroh2),roh1(nroh1),thth(nth),
     &  ve(ip,nroh1,nth,nroh2),ind(nroh1,nth)
        open(98,file='new8-final.dat')
        open(55,status='scratch')
        open(66,status='scratch')
        vcut=15d0/27.2114d0
        pesmin=10000
        do i=1,nroh1
          do j=1,nth
           ind(i,j)=0
            do k=1,nroh2
            read(98,*)mm,ra1,th1,ra2,ve1,ve2,ve3
           if (ra1+ra2.lt.0.1) goto 7
            if (k.eq.1) then
            rra1=ra1
            rra2=ra2
            tth1=th1
            else
            dal=abs(ra1-rra1)+abs(th1-tth1)
            if (dal.gt.0.01) then
            write(*,*)' error  ',ii,ra1,rra1,ra2,rra2
            stop
            endif
            endif
            if (ve1.lt.pesmin) then
            pesmin=ve1
            r1m=ra1
            r2m=ra2
            thm=th1
            endif
           ind(i,j)=ind(i,j)+1
            roh2(i,j,k)=ra2
            roh1(i)=ra1
            thth(j)=th1
            ve(1,i,j,k)=ve1-vmin1
            ve(2,i,j,k)=ve2-vmin1
            ve(3,i,j,k)=ve3-vmin1
            enddo
7          continue
          enddo  
        enddo
        write(55,*)r1m,r2m,thm
c        write(*,*)r1m,r2m,thm,pesmin
c        write(*,*)' th1=',(thth(i),i=1,nth)
c        write(*,*)' roh1=',(roh1(i),i=1,nroh1)
          do i=1,nroh1
          do j=1,nth
           do k=1,ind(i,j)
          if (ve(1,i,j,k).gt.vcut) ve(1,i,j,k)=vcut
          if (ve(2,i,j,k).gt.vcut) ve(2,i,j,k)=vcut
          if (ve(3,i,j,k).gt.vcut) ve(3,i,j,k)=vcut
          write(66,1001)roh1(i),thth(j),roh2(i,j,k)
     &,ve(1,i,j,k),ve(2,i,j,k),ve(3,i,j,k)
 1001      format (3f8.3,6f12.8)
           enddo
         enddo
       enddo
       close(98)
       close(66)
       end

        subroutine h2oadipes(r10,r20,cthi,va,istate)
        implicit real*8(a-h,o-z)
        parameter (rbohr=0.5291771)
        parameter (pi=3.141592653589793d0)
	common /bvcut/vcut
c : r1=r(o-h1), r2=r(o-h2)
        r1=r10
        r2=r20
        cth=cthi
        if (r1.gt.r2) then 
        tmp=r2
        r2=r1
        r1=tmp
        endif
	if(cth.gt.1d0) cth=1d0
	if(cth.lt.-1d0) cth=-1d0
        th=dacos(cth)*180.0d0/pi
        if (r1.lt.1.20d0) r1=1.20d0
        if (r1.gt.3.6d0) r1=3.6d0
        if (r2.lt.1.20d0) r2=1.20d0
	if (r2.gt.16.d0) r2=16.d0
c        if (th.gt.150d0) th=150d0
c        if (th.lt.60d0) th=60d0
        CALL SPl3(r2,th,r1,va,istate)
1002  format(1x,3f12.4,f20.8)
      end


        subroutine h2odiapes(r10,r20,cthi,va,istate)
        implicit real*8(a-h,o-z)
        parameter (rbohr=0.5291771)
        parameter (pi=3.141592653589793d0)
	common /bvcutd/vcut
c : r1=r(o-h1), r2=r(o-h2)
        r1=r10
        r2=r20
        cth=cthi
        if (r1.gt.r2) then 
        tmp=r2
        r2=r1
        r1=tmp
        endif
	if(cth.gt.1d0) cth=1d0
	if(cth.lt.-1d0) cth=-1d0
        th=dacos(cth)*180.0d0/pi
        if (r1.lt.1.20d0) r1=1.20d0
        if (r1.gt.3.6d0) r1=3.6d0
        if (r2.lt.1.20d0) r2=1.20d0
	if (r2.gt.16.d0) r2=16.d0
        CALL SPl3d(r2,th,r1,va,istate)
        if (istate.eq.3) then
          if (va.le.0d0) va=0d0
        endif
1002  format(1x,3f12.4,f20.8)
      end




       subroutine spl3(r1,th,r2,v,istate)
       implicit real*8(a-h,o-z)
       parameter (nroh1=19,nth=19,nroh2=70,ip=3)
	parameter (m=100,n=100,l=100)
	dimension xt(m)
       dimension dty(l),ddty(l),s1(l),ds1(l),dds1(l),h1(l)
       dimension dny(n),ddny(n),s2(l),ds2(l),dds2(l),h2(n)
       dimension dhy(m),ddhy(m),s3(l),ds3(l),dds3(l),h3(m)
       dimension y(m),ss(m),sss(m),y2(l)
       data dy1,dyn/1.0d30,1.0d30/
        common /bvcut/vcut
        common /pes/pesmin,roh2(nroh1,nth,nroh2),roh1(nroh1),thth(nth),
     &  ve(ip,nroh1,nth,nroh2),ind(nroh1,nth)
                     
        do iop=istate,istate
        do 20 i=1,nroh1
       do 10 j=1,nth
	nh=0
       do 2 k=1,ind(i,j)
	nh=nh+1
	xt(nh)=roh2(i,j,k)
	y(nh)=ve(iop,i,j,k)
   2   continue
  	r1a=r1
	call spline(xt,y,nh,dy1,dyn,y2)
        call splint(xt,y,y2,nh,r1a,y3)
 22	continue
	if (y3.gt.vcut) y3=vcut
        ss(j)=y3
   10   continue
	nthth=0
       do 5 j=1,nth
	nthth=nthth+1
	xt(nthth)=thth(j)
        y(nthth)=ss(j)
   5   continue
	call spline(xt,y,nthth,0.0d0,0.0d0,y2)
	call splint(xt,y,y2,nthth,th,yw2)
 33	continue
	if(yw2.gt.vcut) yw2=vcut
       sss(i)=yw2
   20    continue
	nr1=0
c	write(*,*)' vrhh=',(sss(i),i=1,noh)
	do i=1,nroh1
	nr1=nr1+1
	xt(nr1)=roh1(i)
	y(nr1)=sss(i)
  	enddo
	call spline(xt,y,nr1,dy1,dyn,y2)
	call splint(xt,y,y2,nr1,r2,yw2)
   44	continue
	if (yw2.gt.vcut) yw2=vcut
        v=yw2
       enddo
       end




       subroutine spl3d(r1,th,r2,v,istate)
       implicit real*8(a-h,o-z)
       parameter (nroh1=19,nth=19,nroh2=70,ip=4)
	parameter (m=100,n=100,l=100)
	dimension xt(m)
       dimension dty(l),ddty(l),s1(l),ds1(l),dds1(l),h1(l)
       dimension dny(n),ddny(n),s2(l),ds2(l),dds2(l),h2(n)
       dimension dhy(m),ddhy(m),s3(l),ds3(l),dds3(l),h3(m)
       dimension y(m),ss(m),sss(m),y2(l)
       data dy1,dyn/1.0d30,1.0d30/
        common /bvcutd/vcut
        common /pesd/pesmin,roh2(nroh1,nth,nroh2),roh1(nroh1),thth(nth),
     &  ve(ip,nroh1,nth,nroh2),ind(nroh1,nth)
                     
        do iop=istate,istate
        do 20 i=1,nroh1
       do 10 j=1,nth
	nh=0
       do 2 k=1,ind(i,j)
	nh=nh+1
	xt(nh)=roh2(i,j,k)
	y(nh)=ve(iop,i,j,k)
   2   continue
  	r1a=r1
	call spline(xt,y,nh,dy1,dyn,y2)
        call splint(xt,y,y2,nh,r1a,y3)
 22	continue
	if (y3.gt.vcut) y3=vcut
        ss(j)=y3
   10   continue
	nthth=0
       do 5 j=1,nth
	nthth=nthth+1
	xt(nthth)=thth(j)
        y(nthth)=ss(j)
   5   continue
	call spline(xt,y,nthth,0.0d0,0.0d0,y2)
	call splint(xt,y,y2,nthth,th,yw2)
 33	continue
	if(yw2.gt.vcut) yw2=vcut
       sss(i)=yw2
   20    continue
	nr1=0
c	write(*,*)' vrhh=',(sss(i),i=1,noh)
	do i=1,nroh1
	nr1=nr1+1
	xt(nr1)=roh1(i)
	y(nr1)=sss(i)
  	enddo
	call spline(xt,y,nr1,dy1,dyn,y2)
	call splint(xt,y,y2,nr1,r2,yw2)
   44	continue
	if (yw2.gt.vcut) yw2=vcut
        v=yw2
       enddo
       end


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
 1    if (khi-klo.gt.1) then
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
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     *2)/6.0d0
      return
      END
C##############################################################################
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

!-->    irt.eq.0.0d0,accurate renner-teller effect
        subroutine potrealz()
        implicit none
        integer,parameter::nroh2=32,nth=13,nroh1=100,ntmp1=24,ip=2
        integer:: i,j,k,ind(nth,nroh2)
        real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
        real(kind=8):: tmp1(ntmp1),ra1,ra2,th1,ve1,ve2,dal
        real(kind=8):: ve(ip,nth,nroh2,nroh1),rra2,rra1,tth1
         common /peslz/roh2,roh1,thth,ve,ind
        open(20,file='renner-lz.dat')
        do i=1,nth
          do j=1,nroh2
             do k=1,nroh1
              read(20,*)th1,ra2,ra1,ve1,ve2 
              if(k.eq.1) then
              rra2=ra2
              rra1=ra1
              tth1=th1
              else
              dal=abs(ra2-rra2)+abs(th1-tth1)
                 if(dal.gt.0.01) then
                     write(*,*)'error',tth1,rra2,rra1
                     stop
                  endif
              endif
              ind(i,j)=k
              ve(1,i,j,k)=ve1
              ve(2,i,j,k)=ve2
              roh1(i,j,k)=ra1
              roh2(j)=ra2
              thth(i)=th1            
              if(ra1.eq.14.0d0) exit
             enddo
          enddo
        enddo
 
       close(20)
       return
       end        

       subroutine h2olz(cth,r2,r1,v1,nstate)
       implicit none
!-->   r2 is the small r, r1 is the large R
       integer,parameter::nroh2=32,nth=13,nroh1=100,ip=2,ntem=100
       integer:: ind(nth,nroh2),i,j,k,nh,nr2,nthth,nstate
       real(kind=8):: th,cth,r2,r1,v1,v2,v3,vr1,vr2,vr3,vr4,vr5,vr6
       real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
       real(kind=8):: ve(ip,nth,nroh2,nroh1)
       real(kind=8):: xt(ntem),y1(ntem),y2(ntem),y3(ntem),ay1(ntem)
       real(kind=8):: ay2(ntem),ay3(ntem),ss1(ntem),ss2(ntem),ss3(ntem)
       real(kind=8):: by1(ntem),by2(ntem),by3(ntem)       
       real(kind=8):: sss1(ntem),sss2(ntem),sss3(ntem)
       real(kind=8):: cy1(ntem),cy2(ntem),cy3(ntem),pi
       common /peslz/roh2,roh1,thth,ve,ind
       pi=dacos(-1.0d0)
       v1=0d0 
       if(cth.gt.1d0) cth=1d0
       if(cth.lt.-1d0) cth=-1d0
       th=dacos(cth)*180.0d0/pi
       if(th.le.60.0d0) th=60.0d0
       if(r1.le.1.2d0) r1=1.2d0
       if(r1.ge.14.0d0) r1=14.0d0
       if(r2.le.1.2d0) r2=1.2d0
       if(r2.ge.14.0d0) r2=14.0d0
       do i=1,nth
          do j=1,nroh2
             nh=0
             do k=1,ind(i,j)
                nh=nh+1
                xt(nh)=roh1(i,j,k)
                y1(nh)=ve(nstate,i,j,k)
              enddo
              call spline(xt,y1,nh,1.0d30,1.0d30,ay1)
              call splint(xt,y1,ay1,nh,r1,vr1)
              
              if(vr1.le.-1.414d0) vr1=-1.414d0 
              if(vr1.ge.1.414d0) vr1=1.414d0               
              ss1(j)=vr1
          enddo
          nr2=0          
          do j=1,nroh2
             nr2=nr2+1
             xt(nr2)=roh2(j)
             y1(nr2)=ss1(j)
          enddo
               
          call spline(xt,y1,nr2,1.0d30,1.0d30,by1)
          call splint(xt,y1,by1,nr2,r2,vr4)
          
          if(vr4.le.-1.414d0) vr4=-1.414d0
          if(vr4.ge.1.414d0) vr4=1.414d0
          sss1(i)=vr4
       enddo 
       nthth=0 
       do i=1,nth
          nthth=nthth+1
          xt(nthth)=thth(i)
          y1(nthth)=sss1(i)
       enddo

       call spline(xt,y1,nthth,0.0d0,0.0d0,cy1)
       call splint(xt,y1,cy1,nthth,th,v1)
        
       if(v1.le.-1.414d0) v1=-1.414d0
       if(v1.ge.1.414d0) v1=1.414d0
       return
       end 


        subroutine potrealzlz()
        implicit none
        integer,parameter::nroh2=32,nth=13,nroh1=100,ntmp1=24,ip=3
        integer:: i,j,k,ind(nth,nroh2)
        real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
        real(kind=8):: tmp1(ntmp1),ra1,ra2,th1,ve1,ve2,ve3,dal
        real(kind=8):: ve(ip,nth,nroh2,nroh1),rra2,rra1,tth1
         common /peslzlz/roh2,roh1,thth,ve,ind
        open(20,file='renner-lzlz.dat')
        do i=1,nth
          do j=1,nroh2
             do k=1,nroh1
              read(20,*)th1,ra2,ra1,ve1,ve2,ve3 
              if(k.eq.1) then
              rra2=ra2
              rra1=ra1
              tth1=th1
              else
              dal=abs(ra2-rra2)+abs(th1-tth1)
                 if(dal.gt.0.01) then
                     write(*,*)'error',tth1,rra2,rra1
                     stop
                  endif
              endif
              ind(i,j)=k
              ve(1,i,j,k)=ve1
              ve(2,i,j,k)=ve2
              ve(3,i,j,k)=ve3
              roh1(i,j,k)=ra1
              roh2(j)=ra2
              thth(i)=th1            
              if(ra1.eq.14.0d0) exit
             enddo
          enddo
        enddo
 
       close(20)
       return
       end        

       subroutine h2olzlz(cth,r2,r1,v1,nstate)
       implicit none
       integer,parameter::nroh2=32,nth=13,nroh1=100,ip=3,ntem=100
       integer:: ind(nth,nroh2),i,j,k,nh,nr2,nthth,nstate
       real(kind=8):: th,cth,r2,r1,v1,v2,v3,vr1,vr2,vr3,vr4,vr5,vr6
       real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
       real(kind=8):: ve(ip,nth,nroh2,nroh1)
       real(kind=8):: xt(ntem),y1(ntem),y2(ntem),y3(ntem),ay1(ntem)
       real(kind=8):: ay2(ntem),ay3(ntem),ss1(ntem),ss2(ntem),ss3(ntem)
       real(kind=8):: by1(ntem),by2(ntem),by3(ntem)       
       real(kind=8):: sss1(ntem),sss2(ntem),sss3(ntem)
       real(kind=8):: cy1(ntem),cy2(ntem),cy3(ntem),pi
       common /peslzlz/roh2,roh1,thth,ve,ind
       pi=dacos(-1.0d0)
       v1=0d0 
       if(cth.gt.1d0) cth=1d0
       if(cth.lt.-1d0) cth=-1d0
       th=dacos(cth)*180.0d0/pi
       if(th.le.60.0d0) th=60.0d0
       if(r1.le.1.2d0) r1=1.2d0
       if(r1.ge.14.0d0) r1=14.0d0
       if(r2.le.1.2d0) r2=1.2d0
       if(r2.ge.14.0d0) r2=14.0d0
       do i=1,nth
          do j=1,nroh2
             nh=0
             do k=1,ind(i,j)
                nh=nh+1
                xt(nh)=roh1(i,j,k)
                y1(nh)=ve(nstate,i,j,k)
              enddo
              call spline(xt,y1,nh,1.0d30,1.0d30,ay1)
              call splint(xt,y1,ay1,nh,r1,vr1)
              ss1(j)=vr1
          enddo
 
          nr2=0          
          do j=1,nroh2
             nr2=nr2+1
             xt(nr2)=roh2(j)
             y1(nr2)=ss1(j)
          enddo
          call spline(xt,y1,nr2,1.0d30,1.0d30,by1)
          call splint(xt,y1,by1,nr2,r2,vr4)
          sss1(i)=vr4
       enddo 

       nthth=0 
       do i=1,nth
          nthth=nthth+1
          xt(nthth)=thth(i)
          y1(nthth)=sss1(i)
       enddo
       call spline(xt,y1,nthth,0.0d0,0.0d0,cy1)
       call splint(xt,y1,cy1,nthth,th,v1)
       return
       end 

!-->    irt.ne.0.0d0,<1A'|LzLz|1A'> =1.0d0,<1A'|Lz|1A'> =1.0d0
!-->    the same as states 2A' and 1A''

        subroutine potrealz2()
        implicit none
        integer,parameter::nroh2=32,nth=13,nroh1=100,ntmp1=24,ip=2
        integer:: i,j,k,ind(nth,nroh2)
        real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
        real(kind=8):: tmp1(ntmp1),ra1,ra2,th1,ve1,ve2,dal
        real(kind=8):: ve(ip,nth,nroh2,nroh1),rra2,rra1,tth1
         common /peslz2/roh2,roh1,thth,ve,ind
        open(20,file='renner-lz-2.dat')
        do i=1,nth
          do j=1,nroh2
             do k=1,nroh1
              read(20,*)th1,ra2,ra1,ve1,ve2 
              if(k.eq.1) then
              rra2=ra2
              rra1=ra1
              tth1=th1
              else
              dal=abs(ra2-rra2)+abs(th1-tth1)
                 if(dal.gt.0.01) then
                     write(*,*)'error',tth1,rra2,rra1
                     stop
                  endif
              endif
              ind(i,j)=k
              ve(1,i,j,k)=ve1
              ve(2,i,j,k)=ve2
              roh1(i,j,k)=ra1
              roh2(j)=ra2
              thth(i)=th1            
              if(ra1.eq.14.0d0) exit
             enddo
          enddo
        enddo
 
       close(20)
       return
       end        

       subroutine h2olz2(cth,r2,r1,v1,nstate)
       implicit none
!-->   r2 is the small r, r1 is the large R
       integer,parameter::nroh2=32,nth=13,nroh1=100,ip=2,ntem=100
       integer:: ind(nth,nroh2),i,j,k,nh,nr2,nthth,nstate
       real(kind=8):: th,cth,r2,r1,v1,v2,v3,vr1,vr2,vr3,vr4,vr5,vr6
       real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
       real(kind=8):: ve(ip,nth,nroh2,nroh1)
       real(kind=8):: xt(ntem),y1(ntem),y2(ntem),y3(ntem),ay1(ntem)
       real(kind=8):: ay2(ntem),ay3(ntem),ss1(ntem),ss2(ntem),ss3(ntem)
       real(kind=8):: by1(ntem),by2(ntem),by3(ntem)       
       real(kind=8):: sss1(ntem),sss2(ntem),sss3(ntem)
       real(kind=8):: cy1(ntem),cy2(ntem),cy3(ntem),pi
       common /peslz2/roh2,roh1,thth,ve,ind
       pi=dacos(-1.0d0)
       v1=0d0 
       if(cth.gt.1d0) cth=1d0
       if(cth.lt.-1d0) cth=-1d0
       th=dacos(cth)*180.0d0/pi
       if(th.le.60.0d0) th=60.0d0
       if(r1.le.1.2d0) r1=1.2d0
       if(r1.ge.14.0d0) r1=14.0d0
       if(r2.le.1.2d0) r2=1.2d0
       if(r2.ge.14.0d0) r2=14.0d0
       do i=1,nth
          do j=1,nroh2
             nh=0
             do k=1,ind(i,j)
                nh=nh+1
                xt(nh)=roh1(i,j,k)
                y1(nh)=ve(nstate,i,j,k)
              enddo
              call spline(xt,y1,nh,1.0d30,1.0d30,ay1)
              call splint(xt,y1,ay1,nh,r1,vr1)
              
              if(vr1.le.-1.414d0) vr1=-1.414d0 
              if(vr1.ge.1.414d0) vr1=1.414d0               
              ss1(j)=vr1
          enddo
          nr2=0          
          do j=1,nroh2
             nr2=nr2+1
             xt(nr2)=roh2(j)
             y1(nr2)=ss1(j)
          enddo
               
          call spline(xt,y1,nr2,1.0d30,1.0d30,by1)
          call splint(xt,y1,by1,nr2,r2,vr4)
          
          if(vr4.le.-1.414d0) vr4=-1.414d0
          if(vr4.ge.1.414d0) vr4=1.414d0
          sss1(i)=vr4
       enddo 
       nthth=0 
       do i=1,nth
          nthth=nthth+1
          xt(nthth)=thth(i)
          y1(nthth)=sss1(i)
       enddo

       call spline(xt,y1,nthth,0.0d0,0.0d0,cy1)
       call splint(xt,y1,cy1,nthth,th,v1)
        
       if(v1.le.-1.414d0) v1=-1.414d0
       if(v1.ge.1.414d0) v1=1.414d0
       return
       end 


        subroutine potrealzlz2()
        implicit none
        integer,parameter::nroh2=32,nth=13,nroh1=100,ntmp1=24,ip=3
        integer:: i,j,k,ind(nth,nroh2)
        real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
        real(kind=8):: tmp1(ntmp1),ra1,ra2,th1,ve1,ve2,ve3,dal
        real(kind=8):: ve(ip,nth,nroh2,nroh1),rra2,rra1,tth1
         common /peslzlz2/roh2,roh1,thth,ve,ind
        open(20,file='renner-lzlz-2.dat')
        do i=1,nth
          do j=1,nroh2
             do k=1,nroh1
              read(20,*)th1,ra2,ra1,ve1,ve2,ve3 
              if(k.eq.1) then
              rra2=ra2
              rra1=ra1
              tth1=th1
              else
              dal=abs(ra2-rra2)+abs(th1-tth1)
                 if(dal.gt.0.01) then
                     write(*,*)'error',tth1,rra2,rra1
                     stop
                  endif
              endif
              ind(i,j)=k
              ve(1,i,j,k)=ve1
              ve(2,i,j,k)=ve2
              ve(3,i,j,k)=ve3
              roh1(i,j,k)=ra1
              roh2(j)=ra2
              thth(i)=th1            
              if(ra1.eq.14.0d0) exit
             enddo
          enddo
        enddo
 
       close(20)
       return
       end        

       subroutine h2olzlz2(cth,r2,r1,v1,nstate)
       implicit none
       integer,parameter::nroh2=32,nth=13,nroh1=100,ip=3,ntem=100
       integer:: ind(nth,nroh2),i,j,k,nh,nr2,nthth,nstate
       real(kind=8):: th,cth,r2,r1,v1,v2,v3,vr1,vr2,vr3,vr4,vr5,vr6
       real(kind=8):: roh2(nroh2),thth(nth),roh1(nth,nroh2,nroh1)
       real(kind=8):: ve(ip,nth,nroh2,nroh1)
       real(kind=8):: xt(ntem),y1(ntem),y2(ntem),y3(ntem),ay1(ntem)
       real(kind=8):: ay2(ntem),ay3(ntem),ss1(ntem),ss2(ntem),ss3(ntem)
       real(kind=8):: by1(ntem),by2(ntem),by3(ntem)       
       real(kind=8):: sss1(ntem),sss2(ntem),sss3(ntem)
       real(kind=8):: cy1(ntem),cy2(ntem),cy3(ntem),pi
       common /peslzlz2/roh2,roh1,thth,ve,ind
       pi=dacos(-1.0d0)
       v1=0d0 
       if(cth.gt.1d0) cth=1d0
       if(cth.lt.-1d0) cth=-1d0
       th=dacos(cth)*180.0d0/pi
       if(th.le.60.0d0) th=60.0d0
       if(r1.le.1.2d0) r1=1.2d0
       if(r1.ge.14.0d0) r1=14.0d0
       if(r2.le.1.2d0) r2=1.2d0
       if(r2.ge.14.0d0) r2=14.0d0
       do i=1,nth
          do j=1,nroh2
             nh=0
             do k=1,ind(i,j)
                nh=nh+1
                xt(nh)=roh1(i,j,k)
                y1(nh)=ve(nstate,i,j,k)
              enddo
              call spline(xt,y1,nh,1.0d30,1.0d30,ay1)
              call splint(xt,y1,ay1,nh,r1,vr1)
              ss1(j)=vr1
          enddo
 
          nr2=0          
          do j=1,nroh2
             nr2=nr2+1
             xt(nr2)=roh2(j)
             y1(nr2)=ss1(j)
          enddo
          call spline(xt,y1,nr2,1.0d30,1.0d30,by1)
          call splint(xt,y1,by1,nr2,r2,vr4)
          sss1(i)=vr4
       enddo 

       nthth=0 
       do i=1,nth
          nthth=nthth+1
          xt(nthth)=thth(i)
          y1(nthth)=sss1(i)
       enddo
       call spline(xt,y1,nthth,0.0d0,0.0d0,cy1)
       call splint(xt,y1,cy1,nthth,th,v1)
       return
       end 
