

      subroutine krbkrb_lr(RR,r1,r2,th1,th2,phi,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,th1,th2,phi
      real*8,intent(out):: v ! in hartree !
      real*8 RR0,r10,r20
      real*8 U_el,U_ind,U_disp
      real*8 cth1,cth2,sth1,sth2,cphi
      real*8 vec(6)
      real*8,parameter:: PI=dacos(-1.d0)

      RR0=RR; r10=r1; r20=r2
      cth1=dcos(th1*PI/180.d0)
      cth2=dcos(th2*PI/180.d0)
      sth1=dsin(th1*PI/180.d0)
      sth2=dsin(th2*PI/180.d0)
      cphi=dcos(phi*PI/180.d0)
      vec(1)=cth1
      vec(2)=cth2
      vec(3)=cth1*cth2+sth1*sth2*cphi
      vec(4)=-vec(2)
      vec(5)=-vec(1)
      vec(6)=vec(3)

      call krbkrb_lr_el  (RR0,r10,r20,vec,U_el)
      call krbkrb_lr_ind (RR0,r10,r20,vec,U_ind)
      call krbkrb_lr_disp(RR0,r10,r20,vec,U_disp)
      v=U_el+U_ind+U_disp

      return

      endsubroutine


      subroutine krbkrb_lr_el(RR,r1,r2,vec,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,vec(6)
      real*8,intent(out):: v ! in hartree !

      real*8 c1,c2,c3,c4,c5,c6
      real*8 U_dd,U_dq1,U_dq2,U_qq
      real*8 dA,dB,qA,qB
      real*8,external:: krb_dip,krb_quad

      dA=krb_dip(r1)
      dB=krb_dip(r2)
      qA=krb_quad(r1)
      qB=krb_quad(r2)

      c1=vec(1)
      c2=vec(2)
      c3=vec(3)
      c4=vec(4)
      c5=vec(5)
      c6=vec(6)

      U_dd =dA*dB*(c3-3.d0*c1*c2)/RR**3
      U_dq1=1.5d0*dA*qB*(-c1-2.d0*c3*c2+5.d0*c1*c2**2)/RR**4
      U_dq2=1.5d0*dB*qA*(-c4-2.d0*c6*c5+5.d0*c4*c5**2)/RR**4
      U_qq =0.75d0*qA*qB*(1.d0-5.d0*(c1**2+c2**2)+2.d0*c3**2
     &     -20.d0*c1*c2*c3+35.d0*c1**2*c2**2)/RR**5
      v=U_dd+U_dq1+U_dq2+U_qq

      return

      endsubroutine



      subroutine krbkrb_lr_ind(RR,r1,r2,vec,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,vec(6)
      real*8,intent(out):: v ! in hartree !

      real*8 c1,c2,c3
      real*8 term(12)
      real*8 dA,dB,qA,qB,alA,alB,algaA,algaB,ApA,AvA,ApB,AvB
      real*8,external:: krb_dip,krb_quad
      real*8,external:: krb_alpha,krb_alga,krb_AAp,krb_AAv

      dA=krb_dip(r1)
      dB=krb_dip(r2)
      qA=krb_quad(r1)
      qB=krb_quad(r2)
      alA=krb_alpha(r1)
      alB=krb_alpha(r2)
      algaA=krb_alga(r1)
      algaB=krb_alga(r2)
      ApA=krb_AAp(r1)
      ApB=krb_AAp(r2)
      AvA=krb_AAv(r1)
      AvB=krb_AAv(r2)

      c1=vec(1)
      c2=vec(2)
      c3=vec(3)
      term(1)=-0.5d0*dA**2*alB*(1.d0+3.d0*c1**2)/RR**6
      term(2)=-0.5d0*dA**2*algaB*(3.d0*(c3-3.d0*c1*c2)**2
     &        -1.d0-3.d0*c1**2)/RR**6
      term(3)=-1.5d0*dA**2*ApB*(-c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))/RR**7
      term(4)=2.d0*dA**2*AvB*(c2*(1.d0+c1**2)-2.d0*c1*(c3-3.d0*c1*c2)
     &       -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))/RR**7
      term(5)=-6.d0*dA*qA*alB*c1**2/RR**7
      term(6)=3.d0*dA*qA*algaB*(1.5d0*(-c2*(c3-3.d0*c1*c2)
     &       -c1*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &       +2.d0*c1**3)/RR**7

      c1=vec(4)
      c2=vec(5)
      c3=vec(6)
      term(7)=-0.5d0*dB**2*alA*(1.d0+3.d0*c1**2)/RR**6
      term(8)=-0.5d0*dB**2*algaA*(3.d0*(c3-3.d0*c1*c2)**2
     &        -1.d0-3.d0*c1**2)/RR**6
      term(9)=-1.5d0*dB**2*ApA*(-c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))/RR**7
      term(10)=2.d0*dB**2*AvA*(c2*(1.d0+c1**2)-2.d0*c1*(c3-3.d0*c1*c2)
     &       -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))/RR**7
      term(11)=-6.d0*dB*qB*alA*c1**2/RR**7
      term(12)=3.d0*dB*qB*algaA*(1.5d0*(-c2*(c3-3.d0*c1*c2)
     &       -c1*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &       +2.d0*c1**3)/RR**7

      v=sum(term(1:12))

      return

      endsubroutine



      subroutine krbkrb_lr_disp(RR,r1,r2,vec,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,vec(6)
      real*8,intent(out):: v ! in hartree !

      real*8,parameter:: Uex=3.1752303d-2
      real*8 c1,c2,c3,c4,c5,c6
      real*8 term(9)
      real*8 alA,alB,algaA,algaB,gaA,gaB,ApA,AvA,ApB,AvB
      real*8,external:: krb_alpha,krb_alga,krb_AAp,krb_AAv

      alA=krb_alpha(r1)
      alB=krb_alpha(r2)
      algaA=krb_alga(r1)
      algaB=krb_alga(r2)
      gaA=algaA/alA
      gaB=algaB/alB
      ApA=krb_AAp(r1)
      ApB=krb_AAp(r2)
      AvA=krb_AAv(r1)
      AvB=krb_AAv(r2)

      c1=vec(1)
      c2=vec(2)
      c3=vec(3)
      c4=vec(4)
      c5=vec(5)
      c6=vec(6)

      term(1)=-1.5d0*Uex*alA*alB*(1.d0+gaA*0.5d0*(3.d0*c1**2-1.d0)
     &       +gaB*0.5d0*(3.d0*c2**2-1.d0)+1.5d0*gaA*gaB*
     &        ((3.d0*c1*c2-c3)**2-c1**2-c2**2))/RR**6
      term(2)=3.d0*Uex*alA*ApB*c2**3/RR**7
      term(3)=3.d0*Uex*alB*ApA*c5**3/RR**7
      term(4)=2.d0*Uex*alA*AvB*(3.d0*c2-2.d0*c2**3)/RR**7
      term(5)=2.d0*Uex*alB*AvA*(3.d0*c5-2.d0*c5**3)/RR**7
      term(6)=-1.5d0*Uex*algaA*ApB*(1.5d0*(-c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &        +2.d0*c2**3)/RR**7
      term(7)=-1.5d0*Uex*algaB*ApA*(1.5d0*(-c4*(c6-3.d0*c4*c5)
     &        -c5*(2.d0*c6**2-11.d0*c4*c5*c6+15.d0*c4**2*c5**2))
     &        +2.d0*c5**3)/RR**7
      term(8)=-1.5d0*Uex*algaA*AvB*(-2.d0*(c2*(1.d0+c1**2)
     &        -2.d0*c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &        +4.d0/3.d0*(3.d0*c2-2.d0*c2**3))/RR**7
      term(9)=-1.5d0*Uex*algaB*AvA*(-2.d0*(c5*(1.d0+c4**2)
     &        -2.d0*c4*(c6-3.d0*c4*c5)
     &        -c5*(2.d0*c6**2-11.d0*c4*c5*c6+15.d0*c4**2*c5**2))
     &        +4.d0/3.d0*(3.d0*c5-2.d0*c5**3))/RR**7

      v=sum(term(1:9))

      return

      endsubroutine


C--------------- Parametric functions ---------------------------------
      function krb_dip(r)
      implicit none

      real*8 r,krb_dip ! in bohr and hartree !

      integer i
      real*8 a(0:3)
      real*8,parameter:: r0=7.693d0
      real*8,parameter:: autoang=0.529d0
      real*8 d,q,pavg,pdel
      data a /0.2347d0,0.01624d0,-0.00205d0,-0.00309d0/

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call KRb_multipole(r,d,q,pavg,pdel)
      krb_dip=d

      return

      endfunction


      function krb_quad(r)
      implicit none

      real*8 r,krb_quad ! in bohr and hartree !

      integer i
      real*8 a(0:3)
      real*8,parameter:: r0=7.693d0
      real*8,parameter:: autoang=0.529d0
      real*8 d,q,pavg,pdel
      data a /14.9052d0,2.01218d0,-0.45403d0,-0.17310d0/

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call KRb_multipole(r,d,q,pavg,pdel)
      krb_quad=q

      return

      endfunction


      function krb_alpha(r)
      implicit none

      real*8 r,krb_alpha ! in bohr and hartree !

      integer i
      real*8 ra,rb
      real*8 a(0:3)
      real*8,parameter:: r0=7.693d0
      real*8,parameter:: autoang=0.529177249d0
      real*8 d,q,pavg,pdel,p1,p2
      data a /524.129d0,80.5676d0,8.1818d0,-2.4048d0/

      rb=r*autoang
      p1=8.6471*rb**3-122.1*rb**2+501.44*rb-1005.8
      p2=41.764*rb**3-558.71*rb**2+2135.0*rb-3023.7
      krb_alpha=(p1+p1+p2)/3.d0

      return

      endfunction


      function krb_alga(r)
      implicit none

      real*8 r,krb_alga ! in bohr and hartree !

      integer i
      real*8 a(0:3)
      real*8 ra,rb
      real*8,parameter:: r0=7.693d0
      real*8,parameter:: autoang=0.529177249d0
      real*8 d,q,pavg,pdel,p1,p2
      data a /-245.5766d0,-97.8348d0,-7.1062d0,2.555d0/

      rb=r*autoang
      p1=8.6471*rb**3-122.1*rb**2+501.44*rb-1005.8
      p2=41.764*rb**3-558.71*rb**2+2135.0*rb-3023.7
      krb_alga=(p2-p1)/3.d0

      return

      endfunction



      function krb_AAp(r)
      implicit none

      real*8 r,krb_AAp
      real*8 ra,rb
      real*8,parameter:: autoang=0.529177249d0

      rb=r*autoang
      krb_AAp=3214.2*rb**3-36370.0*rb**2+134855.0*rb-166981.0

      return

      endfunction


      function krb_AAv(r)
      implicit none

      real*8 r,krb_AAv

      krb_AAv=0.d0

      return

      endfunction




      subroutine k2rb2_lr(RR,r1,r2,th1,th2,phi,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,th1,th2,phi
      real*8,intent(out):: v ! in hartree !
      real*8 RR0,r10,r20
      real*8 U_el,U_ind,U_disp
      real*8 cth1,cth2,sth1,sth2,cphi
      real*8 vec(6)
      real*8,parameter:: PI=dacos(-1.d0)

      RR0=RR; r10=r1; r20=r2
      cth1=dcos(th1*PI/180.d0)
      cth2=dcos(th2*PI/180.d0)
      sth1=dsin(th1*PI/180.d0)
      sth2=dsin(th2*PI/180.d0)
      cphi=dcos(phi*PI/180.d0)
      vec(1)=cth1
      vec(2)=cth2
      vec(3)=cth1*cth2+sth1*sth2*cphi
      vec(4)=-vec(2)
      vec(5)=-vec(1)
      vec(6)=vec(3)

      call k2rb2_lr_el  (RR0,r10,r20,vec,U_el)
      call k2rb2_lr_disp(RR0,r10,r20,vec,U_disp)
      v=U_el+U_disp

      return
      endsubroutine



      subroutine k2rb2_lr_el(RR,r1,r2,vec,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,vec(6)
      real*8,intent(out):: v ! in hartree !

      real*8 c1,c2,c3,c4,c5,c6
      real*8 U_dd,U_dq1,U_dq2,U_qq
      real*8 qA,qB
      real*8,external:: k2_quad,rb2_quad

      qA=k2_quad(r1)
      qB=rb2_quad(r2)

      c1=vec(1)
      c2=vec(2)
      c3=vec(3)
      c4=vec(4)
      c5=vec(5)
      c6=vec(6)

      U_qq =0.75d0*qA*qB*(1.d0-5.d0*(c1**2+c2**2)+2.d0*c3**2
     &     -20.d0*c1*c2*c3+35.d0*c1**2*c2**2)/RR**5
      v=U_qq

      return
      endsubroutine



      subroutine k2rb2_lr_disp(RR,r1,r2,vec,v)
      implicit none

      real*8,intent(in):: RR,r1,r2,vec(6)
      real*8,intent(out):: v ! in hartree !

      real*8,parameter:: U1=14135.6d0/219474.63067d0
      real*8,parameter:: U2=3.1752303d-2*2d0
      real*8,parameter:: Uex=(U1*U2)/(U1+U2)
      real*8 c1,c2,c3,c4,c5,c6
      real*8 term(9)
      real*8 alA,alB,algaA,algaB,gaA,gaB,ApA,AvA,ApB,AvB
      real*8,external:: k2_alpha,k2_alga,k2_AAp,k2_AAv
      real*8,external:: rb2_alpha,rb2_alga,rb2_AAp,rb2_AAv

      alA=k2_alpha(r1)
      alB=rb2_alpha(r2)
      algaA=k2_alga(r1)
      algaB=rb2_alga(r2)
      gaA=algaA/alA
      gaB=algaB/alB
      ApA=k2_AAp(r1)
      ApB=rb2_AAp(r2)
      AvA=k2_AAv(r1)
      AvB=rb2_AAv(r2)

      c1=vec(1)
      c2=vec(2)
      c3=vec(3)
      c4=vec(4)
      c5=vec(5)
      c6=vec(6)

      term(1)=-1.5d0*Uex*alA*alB*(1.d0+gaA*0.5d0*(3.d0*c1**2-1.d0)
     &       +gaB*0.5d0*(3.d0*c2**2-1.d0)+1.5d0*gaA*gaB*
     &        ((3.d0*c1*c2-c3)**2-c1**2-c2**2))/RR**6
      term(2)=3.d0*Uex*alA*ApB*c2**3/RR**7
      term(3)=3.d0*Uex*alB*ApA*c5**3/RR**7
      term(4)=2.d0*Uex*alA*AvB*(3.d0*c2-2.d0*c2**3)/RR**7
      term(5)=2.d0*Uex*alB*AvA*(3.d0*c5-2.d0*c5**3)/RR**7
      term(6)=-1.5d0*Uex*algaA*ApB*(1.5d0*(-c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &        +2.d0*c2**3)/RR**7
      term(7)=-1.5d0*Uex*algaB*ApA*(1.5d0*(-c4*(c6-3.d0*c4*c5)
     &        -c5*(2.d0*c6**2-11.d0*c4*c5*c6+15.d0*c4**2*c5**2))
     &        +2.d0*c5**3)/RR**7
      term(8)=-1.5d0*Uex*algaA*AvB*(-2.d0*(c2*(1.d0+c1**2)
     &        -2.d0*c1*(c3-3.d0*c1*c2)
     &        -c2*(2.d0*c3**2-11.d0*c1*c2*c3+15.d0*c1**2*c2**2))
     &        +4.d0/3.d0*(3.d0*c2-2.d0*c2**3))/RR**7
      term(9)=-1.5d0*Uex*algaB*AvA*(-2.d0*(c5*(1.d0+c4**2)
     &        -2.d0*c4*(c6-3.d0*c4*c5)
     &        -c5*(2.d0*c6**2-11.d0*c4*c5*c6+15.d0*c4**2*c5**2))
     &        +4.d0/3.d0*(3.d0*c5-2.d0*c5**3))/RR**7

      v=sum(term(1:9))

      return
      endsubroutine


C--------------- Parametric functions ---------------------------------
      function k2_quad(r)
      implicit none

      real*8 r,k2_quad ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(r,x,q,x,pavg,x,pdel,x)
      k2_quad=q

      return
      endfunction


      function rb2_quad(r)
      implicit none

      real*8 r,rb2_quad ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(x,r,x,q,x,pavg,x,pdel)
      rb2_quad=q

      return
      endfunction


      function k2_alpha(r)
      implicit none

      real*8 r,k2_alpha ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529177249d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(r,x,q,x,pavg,x,pdel,x)
      k2_alpha=pavg

      return
      endfunction


      function rb2_alpha(r)
      implicit none

      real*8 r,rb2_alpha ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529177249d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(x,r,x,q,x,pavg,x,pdel)
      rb2_alpha=pavg

      return
      endfunction


      function k2_alga(r)
      implicit none

      real*8 r,k2_alga ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529177249d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(r,x,q,x,pavg,x,pdel,x)
      k2_alga=pdel/3.d0

      return
      endfunction


      function rb2_alga(r)
      implicit none

      real*8 r,rb2_alga ! in bohr and hartree !

      integer i
      real*8,parameter:: autoang=0.529177249d0
      real*8 q,pavg,pdel,x

      if(r.lt.3.d0/autoang) stop 'r is too small in dipole!'
      if(r.gt.5.25d0/autoang) stop 'r is too large in dipole!'
      call K2Rb2_multipole(x,r,x,q,x,pavg,x,pdel)
      rb2_alga=pdel/3.d0

      return
      endfunction


      function k2_AAp(r)
      implicit none
      real*8 r,k2_AAp
      k2_AAp=0.d0
      return
      endfunction

      function rb2_AAp(r)
      implicit none
      real*8 r,rb2_AAp
      rb2_AAp=0.d0
      return
      endfunction

      function k2_AAv(r)
      implicit none
      real*8 r,k2_AAv
      k2_AAv=0.d0
      return
      endfunction

      function rb2_AAv(r)
      implicit none
      real*8 r,rb2_AAv
      rb2_AAv=0.d0
      return
      endfunction

        subroutine KRb_multipole(r,d,q,pavg,pdel)
        implicit none
        integer init3
        integer,parameter :: n=46
        real*8,dimension(n) :: x,ydip,yquad,ypolz,ypolx
        real*8,dimension(n) :: ydip0,yquad0,ypolz0,ypolx0
        common /KRb_data/x,ydip,ydip0,yquad,yquad0,ypolz,ypolz0,
     &          ypolx,ypolx0
        real*8,intent(in) :: r
        real*8,intent(out):: d,q,pdel,pavg
        real*8 rkrb,p1,p2
        real*8,parameter :: bohr=0.529177249d0

        data init3/0/
        save init3
        if(init3==0) then
           call KRb_multipole_init()
           init3=1
        endif

        rkrb=r*bohr
        call splint(x,ydip0,ydip,n,rkrb,d)
        call splint(x,yquad0,yquad,n,rkrb,q)
        call splint(x,ypolz0,ypolz,n,rkrb,p1)
        call splint(x,ypolx0,ypolx,n,rkrb,p2)
        pavg=(p1+p2*2.d0)/3.d0
        pdel=p1-p2
        return
        end subroutine

        subroutine KRb_multipole_init()
        implicit none
        integer i
        integer,parameter :: n=46
        real*8,dimension(n) :: x,ydip,yquad,ypolz,ypolx
        real*8,dimension(n) :: ydip0,yquad0,ypolz0,ypolx0
        real*8 dy1,dyn
        common /KRb_data/x,ydip,ydip0,yquad,yquad0,ypolz,ypolz0,
     &          ypolx,ypolx0
        data dy1,dyn/1d30,1d30/

        open(50,file='data_KRb.inp',position='rewind')
        do i=1,n
        read(50,*)x(i),ydip0(i),ypolz0(i),ypolx0(i),yquad0(i)
        enddo
        close(50)
        call spline(x,ydip0,n,dy1,dyn,ydip)
        call spline(x,yquad0,n,dy1,dyn,yquad)
        call spline(x,ypolz0,n,dy1,dyn,ypolz)
        call spline(x,ypolx0,n,dy1,dyn,ypolx)
        return
        end subroutine KRb_multipole_init

        subroutine K2Rb2_multipole(r1,r2,q1,q2,pavg1,pavg2,pdel1,pdel2)
        implicit none
        integer init4
        integer,parameter :: n1=50,n2=49
        real*8,dimension(n1) :: x1,yq1,ypz1,ypx1,yq01,ypz01,ypx01
        real*8,dimension(n2) :: x2,yq2,ypz2,ypx2,yq02,ypz02,ypx02
        common /K2Rb2_data/x1,x2,yq1,yq2,ypz1,ypz2,ypx1,ypx2,
     &         yq01,yq02,ypz01,ypz02,ypx01,ypx02
        real*8,intent(in) :: r1,r2
        real*8,intent(out):: q1,q2,pdel1,pdel2,pavg1,pavg2
        real*8 rk2,rrb2,px1,px2,pz1,pz2
        real*8,parameter :: bohr=0.529177249d0

        data init4/0/
        save init4
        if(init4==0) then
           call K2Rb2_multipole_init()
           init4=1
        endif

        rk2=r1*bohr
        rrb2=r2*bohr
        call splint(x1,yq01,yq1,n1,rk2,q1)
        call splint(x1,ypz01,ypz1,n1,rk2,pz1)
        call splint(x1,ypx01,ypx1,n1,rk2,px1)
        call splint(x2,yq02,yq2,n2,rrb2,q2)
        call splint(x2,ypz02,ypz2,n2,rrb2,pz2)
        call splint(x2,ypx02,ypx2,n2,rrb2,px2)
        pavg1=(pz1+px1*2.d0)/3.d0
        pdel1=pz1-px1
        pavg2=(pz2+px2*2.d0)/3.d0
        pdel2=pz2-px2
        return
        end subroutine

        subroutine K2Rb2_multipole_init()
        implicit none
        integer i
        integer,parameter :: n1=50,n2=49
        real*8,dimension(n1) :: x1,yq1,ypz1,ypx1,yq01,ypz01,ypx01
        real*8,dimension(n2) :: x2,yq2,ypz2,ypx2,yq02,ypz02,ypx02
        common /K2Rb2_data/x1,x2,yq1,yq2,ypz1,ypz2,ypx1,ypx2,
     &         yq01,yq02,ypz01,ypz02,ypx01,ypx02
        real*8 dy1,dyn,a
        data dy1,dyn/1d30,1d30/

        open(51,file='data_K2.inp',position='rewind')
        open(52,file='data_Rb2.inp',position='rewind')
        do i=1,n1
        read(51,*)x1(i),a,ypz01(i),ypx01(i),yq01(i)
        enddo
        do i=1,n2
        read(52,*)x2(i),a,ypz02(i),ypx02(i),yq02(i)
        enddo
        close(51);close(52)
        call spline(x1,yq01,n1,dy1,dyn,yq1)
        call spline(x1,ypz01,n1,dy1,dyn,ypz1)
        call spline(x1,ypx01,n1,dy1,dyn,ypx1)
        call spline(x2,yq02,n2,dy1,dyn,yq2)
        call spline(x2,ypz02,n2,dy1,dyn,ypz2)
        call spline(x2,ypx01,n2,dy1,dyn,ypx2)
        return
        end subroutine K2Rb2_multipole_init


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
C###################################################################
