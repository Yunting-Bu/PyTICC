
        subroutine potential_AB(r,V) ! H2 bond & energy in a.u. !
        implicit none

        real*8,parameter::autoang=0.529177249d0
        real*8,parameter::autocm=219474.63137d0
        real*8,parameter::autoev=27.21138d0

        real*8,parameter::re=0.7414d0
        real*8,parameter::De=4.747d0/autoev
        real*8,parameter::a1=3.961d0
        real*8,parameter::a2=4.064d0
        real*8,parameter::a3=3.574d0

        real*8 p,r,r_ang,V

        r_ang=r*autoang
        p=r_ang-re
        v=-De*(1d0+a1*p+a2*p**2+a3*p**3)*dexp(-a1*p)
        return
        endsubroutine


        include 'interaction-PES.f'

        subroutine interaction_potential_ABC(RR,r,th,v)
        use parameters, only : m_A,m_B,m_C 
        implicit none

        real*8,intent(in):: RR,r,th  ! in au and degree !
        real*8,intent(out):: v  ! in au !
        real*8 RRR,rH2,theta,Vint

        call trans_iso(RR,r,th,RRR,rH2,theta)
!        rH2=1.448739d0;theta=180.d0
!        do RRR=5.d0,10.d0,0.1d0
        call PESH2_Ar(RRR,rH2,theta,Vint,0)
        v=Vint
!        write(*,*)RR,r,th
!        write(*,*)RRR,rH2,theta,Vint*219474.63137d0
!        enddo
!        stop

        return
        endsubroutine


        subroutine trans_iso(RR0,r0,th0,RR1,r1,th1)
        use parameters, only : PI,m_A,m_B,m_C
        implicit none

        real*8,intent(in):: RR0,r0,th0
        real*8,intent(out):: RR1,r1,th1

        real*8 r2,r3,r4,calpha,cth1

        r2=m_A*r0/(m_A+m_B)!!!!!!definition of theta??
        r3=dsqrt(RR0**2+r2**2-2.d0*RR0*r2*dcos(th0*PI/180.d0))
        calpha=(r3**2+r2**2-RR0**2)/(2.d0*r3*r2)
        calpha=max(calpha,-1.d0);calpha=min(calpha,1.d0)

        r4=r0/2.d0
        RR1=dsqrt(r3**2+r4**2-2.d0*r3*r4*calpha)
        cth1=(RR1**2+r4**2-r3**2)/(2.d0*RR1*r4)
        cth1=max(cth1,-1.d0);cth1=min(cth1,1.d0)

        th1=dacos(cth1)*180.d0/PI;r1=r0

        endsubroutine
