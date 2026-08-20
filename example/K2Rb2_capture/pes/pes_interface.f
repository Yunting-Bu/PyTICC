
      include 'dimer_Eab.f'
      include 'krbkrb_lr.f'


      subroutine potential_AB(r,v) ! bond & energy in a.u. !
      implicit none

      real*8 r,v,rang
      real*8,external:: v_krb
      real*8,parameter:: autoang=0.529177249d0

      rang=r*autoang
      v=v_krb(rang)+52.3903382831d0

      return
      end subroutine


      subroutine potential_CD(r,v) ! bond & energy in a.u. !
      implicit none

      real*8 r,v,rang
      real*8,external:: v_krb
      real*8,parameter:: autoang=0.529177249d0

      rang=r*autoang
      v=v_krb(rang)+52.3903382831d0

      return
      end subroutine


      subroutine interaction_potential_ABCD(jcb,v)
      implicit none

      real*8,intent(in):: jcb(6)! in au and degree !
      real*8,intent(out):: v  ! in au !
      real*8 r1,r2,rr,th1,th2,phi,vint

      rr=jcb(1); r1=jcb(2); r2=jcb(3)
      th1=jcb(4); th2=jcb(5); phi=jcb(6)
      call krbkrb_lr(rr,r1,r2,th1,th2,phi,vint)
      v=1.d0*vint

      return
      endsubroutine



