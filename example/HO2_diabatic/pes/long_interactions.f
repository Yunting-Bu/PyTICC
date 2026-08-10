
        module parameters_long
        implicit none

        real*8,parameter :: pi=dacos(-1.d0)
        real*8,parameter :: degtorad=pi/1.8d2
        real*8,parameter :: autoev=27.2113845d0
        real*8,parameter :: autocm=219474.63137d0

        endmodule

        module variates_for_calculation
        implicit none

        integer symmetryA,symmetryB

        real*8 R
        real*8 alphaA,betaA,gammaA
        real*8 alphaB,betaB,gammaB
        real*8 alphaAreg,betaAreg,gammaAreg
        real*8 alphaBreg,betaBreg,gammaBreg

        real*8 rotA(3,3),rotB(3,3)

        real*8 T1(3),T2(3,3),T3(3,3,3),T4(3,3,3,3),T5(3,3,3,3,3)
        real*8 qA
        real*8 dmA_MF(3),qmA_MF(3,3),omA_MF(3,3,3),hmA_MF(3,3,3,3)
        real*8 alphaA_MF(3,3),AA_MF(3,3,3),CA_MF(3,3,3,3),EA_MF(3,3,3,3)
        real*8 qB
        real*8 dmB_MF(3),qmB_MF(3,3),omB_MF(3,3,3),hmB_MF(3,3,3,3)
        real*8 alphaB_MF(3,3),AB_MF(3,3,3),CB_MF(3,3,3,3),EB_MF(3,3,3,3)

        real*8 dmA_DF(3),qmA_DF(3,3),omA_DF(3,3,3),hmA_DF(3,3,3,3)
        real*8 alphaA_DF(3,3),AA_DF(3,3,3),CA_DF(3,3,3,3),EA_DF(3,3,3,3)
        real*8 dmB_DF(3),qmB_DF(3,3),omB_DF(3,3,3),hmB_DF(3,3,3,3)
        real*8 alphaB_DF(3,3),AB_DF(3,3,3),CB_DF(3,3,3,3),EB_DF(3,3,3,3)

        integer judisp
        real*8 UA,UB
        real*8 dispdddd_MF(3,3,3,3)
        real*8 dispdddd_DF(3,3,3,3)

        real*8 U_disp_6,U_disp_7,U_disp_8
        real*8 U_el,U_ind_A,U_ind_B,U_disp,U_int

        contains

        subroutine long_range_interaction
        use parameters_long,only : autocm
        implicit none

        call T
        call eulerrotA
        call eulerrotB
        call transelectA
        call transelectB

        if(judisp==1)then
           call trans_disp6
        endif

        call lrel
        call lrindA
        call lrindB
        call lrdisp
        
        U_el=U_el*autocm
        U_ind_A=U_ind_A*autocm
        U_ind_B=U_ind_B*autocm
        U_disp=U_disp*autocm
        U_disp_6=U_disp_6*autocm
        U_disp_7=U_disp_7*autocm
        U_disp_8=U_disp_8*autocm
        U_int=U_el+U_ind_A+U_ind_B+U_disp

        endsubroutine

        endmodule


        subroutine eulerrotA
        use variates_for_calculation,only : rotA,alphaA,
     &                                      betaA,gammaA,
     &                                      alphaAreg,betaAreg,
     &                                      gammaAreg
        use parameters_long,only : degtorad
        implicit none
        real*8 cosa,sina,cosb,sinb,cosc,sinc

        alphaAreg = alphaA * degtorad
        betaAreg  = betaA  * degtorad
        gammaAreg = gammaA * degtorad

        cosa=dcos(alphaAreg)
        sina=dsin(alphaAreg)
        cosb=dcos(betaAreg)
        sinb=dsin(betaAreg)
        cosc=dcos(gammaAreg)
        sinc=dsin(gammaAreg)

        rotA(1,1) = cosb*cosa*cosc-sina*sinc
        rotA(2,1) = cosb*sina*cosc+cosa*sinc
        rotA(3,1) = -sinb*cosc
        rotA(1,2) = -cosb*cosa*sinc-sina*cosc
        rotA(2,2) = -cosb*sina*sinc+cosa*cosc
        rotA(3,2) = sinb*sinc
        rotA(1,3) = sinb*cosa
        rotA(2,3) = sinb*sina
        rotA(3,3) = cosb

        return
        endsubroutine


        subroutine eulerrotB
        use variates_for_calculation,only : rotB,alphaB,
     &                                      betaB,gammaB,
     &                                      alphaBreg,betaBreg,
     &                                      gammaBreg
        use parameters_long,only : degtorad
        implicit none
        real*8 cosa,sina,cosb,sinb,cosc,sinc

        alphaBreg = alphaB * degtorad
        betaBreg  = betaB  * degtorad
        gammaBreg = gammaB * degtorad

        cosa=dcos(alphaBreg)
        sina=dsin(alphaBreg)
        cosb=dcos(betaBreg)
        sinb=dsin(betaBreg)
        cosc=dcos(gammaBreg)
        sinc=dsin(gammaBreg)

        rotB(1,1) = cosb*cosa*cosc-sina*sinc
        rotB(2,1) = cosb*sina*cosc+cosa*sinc
        rotB(3,1) = -sinb*cosc
        rotB(1,2) = -cosb*cosa*sinc-sina*cosc
        rotB(2,2) = -cosb*sina*sinc+cosa*cosc
        rotB(3,2) = sinb*sinc
        rotB(1,3) = sinb*cosa
        rotB(2,3) = sinb*sina
        rotB(3,3) = cosb

        return
        endsubroutine


        subroutine T
        use variates_for_calculation,only : R,T1,T2,T3,T4,T5
        implicit none
        integer i,j,k,l,m
        real*8 RR(3),delta(3,3)

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


        subroutine transelectA
        use variates_for_calculation,only : rotA,
     &                                      dmA_MF,qmA_MF,omA_MF,hmA_MF,
     &                                      alphaA_MF,AA_MF,CA_MF,EA_MF,
     &                                      dmA_DF,qmA_DF,omA_DF,hmA_DF,
     &                                      alphaA_DF,AA_DF,CA_DF,EA_DF
        implicit none
        integer i,j,k,l,m,n,o,p,s,t

        dmA_DF=0.d0;qmA_DF=0.d0;alphaA_DF=0.d0;omA_DF=0.d0;AA_DF=0.d0
        hmA_DF=0.d0;CA_DF=0.d0;EA_DF=0.d0

        do i=1,3,1
        do j=1,3,1
           dmA_DF(i)=dmA_DF(i)+rotA(i,j)*dmA_MF(j)
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           qmA_DF(i,j)=qmA_DF(i,j)+rotA(i,k)*rotA(j,l)*qmA_MF(k,l)
           alphaA_DF(i,j)=alphaA_DF(i,j)+rotA(i,k)*rotA(j,l)
     &                    *alphaA_MF(k,l)
        enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           omA_DF(i,j,k)=omA_DF(i,j,k)+rotA(i,l)*rotA(j,m)*rotA(k,n)
     &                   *omA_MF(l,m,n)
           AA_DF(i,j,k)=AA_DF(i,j,k)+rotA(i,l)*rotA(j,m)*rotA(k,n)
     &                  *AA_MF(l,m,n)
        enddo;enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
        do o=1,3,1
        do p=1,3,1
           hmA_DF(i,j,k,l)=hmA_DF(i,j,k,l)+rotA(i,m)*rotA(j,n)*rotA(k,o)
     &                     *rotA(l,p)*hmA_MF(m,n,o,p)
           CA_DF(i,j,k,l)=CA_DF(i,j,k,l)+rotA(i,m)*rotA(j,n)*rotA(k,o)
     &                    *rotA(l,p)*CA_MF(m,n,o,p)
           EA_DF(i,j,k,l)=EA_DF(i,j,k,l)+rotA(i,m)*rotA(j,n)*rotA(k,o)
     &                    *rotA(l,p)*EA_MF(m,n,o,p)
        enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo

        return
        endsubroutine


        subroutine transelectB
        use variates_for_calculation,only : rotB,
     &                                      dmB_MF,qmB_MF,omB_MF,hmB_MF,
     &                                      alphaB_MF,AB_MF,CB_MF,EB_MF,
     &                                      dmB_DF,qmB_DF,omB_DF,hmB_DF,
     &                                      alphaB_DF,AB_DF,CB_DF,EB_DF
        implicit none
        integer i,j,k,l,m,n,o,p,s,t

        dmB_DF=0.d0;qmB_DF=0.d0;alphaB_DF=0.d0;omB_DF=0.d0;AB_DF=0.d0
        hmB_DF=0.d0;CB_DF=0.d0;EB_DF=0.d0

        do i=1,3,1
        do j=1,3,1
           dmB_DF(i)=dmB_DF(i)+rotB(i,j)*dmB_MF(j)
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           qmB_DF(i,j)=qmB_DF(i,j)+rotB(i,k)*rotB(j,l)*qmB_MF(k,l)
           alphaB_DF(i,j)=alphaB_DF(i,j)+rotB(i,k)*rotB(j,l)
     &                    *alphaB_MF(k,l)
        enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           omB_DF(i,j,k)=omB_DF(i,j,k)+rotB(i,l)*rotB(j,m)*rotB(k,n)
     &                   *omB_MF(l,m,n)
           AB_DF(i,j,k)=AB_DF(i,j,k)+rotB(i,l)*rotB(j,m)*rotB(k,n)
     &                  *AB_MF(l,m,n)
        enddo;enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
        do o=1,3,1
        do p=1,3,1
           hmB_DF(i,j,k,l)=hmB_DF(i,j,k,l)+rotB(i,m)*rotB(j,n)*rotB(k,o)
     &                     *rotB(l,p)*hmB_MF(m,n,o,p)
           CB_DF(i,j,k,l)=CB_DF(i,j,k,l)+rotB(i,m)*rotB(j,n)*rotB(k,o)
     &                    *rotB(l,p)*CB_MF(m,n,o,p)
           EB_DF(i,j,k,l)=EB_DF(i,j,k,l)+rotB(i,m)*rotB(j,n)*rotB(k,o)
     &                    *rotB(l,p)*EB_MF(m,n,o,p)
        enddo;enddo;enddo;enddo;enddo;enddo;enddo;enddo

        return
        endsubroutine


        subroutine trans_disp6
        use variates_for_calculation,only : rotA,rotB,
     &                                      dispdddd_MF,dispdddd_DF
        implicit none
        integer i,j,k,l,m,n,o,p,q,r,s,t

        dispdddd_DF=0.d0

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
        do o=1,3,1
        do p=1,3,1
           dispdddd_DF(i,j,k,l)=dispdddd_DF(i,j,k,l)
     &                          +rotA(i,m)*rotA(j,n)*rotB(k,o)*rotB(l,p)
     &                          *dispdddd_MF(m,n,o,p)
        enddo;enddo;enddo;enddo
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine lrel
        use variates_for_calculation,only : R,T1,T2,T3,T4,T5,qA,qB,
     &                                      dmA_DF,qmA_DF,omA_DF,hmA_DF,
     &                                      dmB_DF,qmB_DF,omB_DF,hmB_DF,
     &                                      U_el
        implicit none
        integer i,j,k,l,m
        real*8 term(19)

        U_el=0.d0
        term=0.d0

        term(1)=qA*qB/R

        do i=1,3,1
           term(2)=term(2)+T1(i)*qA*dmB_DF(i)
           term(3)=term(3)-T1(i)*dmA_DF(i)*qB
        enddo

        do i=1,3,1
        do j=1,3,1
           term(4)=term(4)+T2(i,j)*qA*qmB_DF(i,j)/3.d0
           term(5)=term(5)-T2(i,j)*dmA_DF(i)*dmB_DF(j)
           term(6)=term(6)+T2(i,j)*qmA_DF(i,j)*qB/3.d0
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           term(7)=term(7)+T3(i,j,k)*qA*omB_DF(i,j,k)/15.d0
           term(8)=term(8)-T3(i,j,k)*dmA_DF(i)*qmB_DF(j,k)/3.d0
           term(9)=term(9)+T3(i,j,k)*qmA_DF(i,j)*dmB_DF(k)/3.d0
           term(10)=term(10)-T3(i,j,k)*omA_DF(i,j,k)*qB/15.d0
        enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(11)=term(11)+T4(i,j,k,l)*qA*hmB_DF(i,j,k,l)/105.d0
           term(12)=term(12)-T4(i,j,k,l)*dmA_DF(i)*omB_DF(j,k,l)/15.d0
           term(13)=term(13)+T4(i,j,k,l)*qmA_DF(i,j)*qmB_DF(k,l)/9.d0
           term(14)=term(14)-T4(i,j,k,l)*omA_DF(i,j,k)*dmB_DF(l)/15.d0
           term(15)=term(15)+T4(i,j,k,l)*hmA_DF(i,j,k,l)*qB/105.d0
        enddo;enddo;enddo;enddo

        if(qA>1.d-8.or.qA<-1.d-8.or.qB>1.d-8.or.qB<-1.d-8)then
           goto 111
        endif

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(16)=term(16)-T5(i,j,k,l,m)*dmA_DF(i)*hmB_DF(j,k,l,m)
     &              /105.d0
           term(17)=term(17)+T5(i,j,k,l,m)*qmA_DF(i,j)*omB_DF(k,l,m)
     &              /45.d0
           term(18)=term(18)-T5(i,j,k,l,m)*omA_DF(i,j,k)*qmB_DF(l,m)
     &              /45.d0
           term(19)=term(19)+T5(i,j,k,l,m)*hmA_DF(i,j,k,l)*dmB_DF(m)/105.d0
        enddo;enddo;enddo;enddo;enddo

        !term(:)=term(:)*2.1947463137d5
111     U_el=sum(term(1:19))
        !write(*,'(f15.10)')U_el*219474.6
        !write(*,'(7f15.5)')term(1),sum(term(2:3)),sum(term(4:6)),
!     !&            sum(term(7:10)),sum(term(11:15)),sum(term(16:21)),U_el
        return
        endsubroutine



        subroutine lrindA
        use variates_for_calculation,only : T1,T2,T3,T4,T5,qB,
     &                                      dmB_DF,qmB_DF,omB_DF,hmB_DF,
     &                                      alphaA_DF,AA_DF,CA_DF,EA_DF,
     &                                      U_ind_A
        implicit none
        integer i,j,k,l,m,n
        real*8 term(17)

        U_ind_A=0.d0
        term=0.d0

        do i=1,3,1
        do j=1,3,1
           term(1)=term(1)-T1(i)*T1(j)*qB*qB*alphaA_DF(i,j)/2.d0
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           term(2)=term(2)-T1(i)*T2(j,k)*qB*dmB_DF(k)*alphaA_DF(i,j)
           term(3)=term(3)+T1(i)*T2(j,k)*qB*qB*AA_DF(i,j,k)/3.d0
        enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(4)=term(4)-T1(i)*T3(j,k,l)*qB*qmB_DF(k,l)
     &             *alphaA_DF(i,j)/3.d0
           term(5)=term(5)-T2(i,j)*T2(k,l)*dmB_DF(j)*dmB_DF(l)
     &             *alphaA_DF(i,k)/2.d0
           term(6)=term(6)+T1(i)*T3(j,k,l)*qB*dmB_DF(l)
     &             *AA_DF(i,j,k)/3.d0
           term(7)=term(7)+T2(i,j)*T2(k,l)*dmB_DF(j)*qB
     &             *AA_DF(i,k,l)/3.d0
           term(8)=term(8)-T2(i,j)*T2(k,l)*qB*qB*CA_DF(i,j,k,l)/6.d0
           term(9)=term(9)-T1(i)*T3(j,k,l)*qB*qB*EA_DF(i,j,k,l)/15.d0
        enddo;enddo;enddo;enddo

        if(qB>1.d-8.or.qB<-1.d-8)then
           goto 222
        endif

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(10)=term(10)-T2(i,j)*T3(k,l,m)*dmB_DF(j)*qmB_DF(l,m)
     &              *alphaA_DF(i,k)/3.d0
           term(11)=term(11)+T2(i,j)*T3(k,l,m)*dmB_DF(j)*dmB_DF(m)
     &              *AA_DF(i,k,l)/3.d0
        enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           term(12)=term(12)-T2(i,j)*T4(k,l,m,n)*dmB_DF(j)*omB_DF(l,m,n)
     &              *alphaA_DF(i,k)/15.d0
           term(13)=term(13)-T3(i,j,k)*T3(l,m,n)*qmB_DF(j,k)*qmB_DF(m,n)
     &              *alphaA_DF(i,l)/18.d0
           term(14)=term(14)+T2(i,j)*T4(k,l,m,n)*dmB_DF(j)*qmB_DF(m,n)
     &              *AA_DF(i,k,l)/9.d0
           term(15)=term(15)+T3(i,j,k)*T3(l,m,n)*qmB_DF(j,k)*dmB_DF(n)
     &              *AA_DF(i,l,m)/9.d0
           term(16)=term(16)-T3(i,j,k)*T3(l,m,n)*dmB_DF(k)*dmB_DF(n)
     &              *CA_DF(i,j,l,m)/6.d0
           term(17)=term(17)-T2(i,j)*T4(k,l,m,n)*dmB_DF(j)*dmB_DF(n)
     &              *EA_DF(i,k,l,m)/15.d0
        enddo;enddo;enddo;enddo;enddo;enddo
        
        !term(:)=term(:)*2.1947463137d5
222     U_ind_A=sum(term(1:17))
        !write(*,'(f15.10)')U_ind_A*219474.6
        !write(*,'(12f15.5)')term(18:29)
        return
        endsubroutine


        
        subroutine lrindB
        use variates_for_calculation,only : T1,T2,T3,T4,T5,qA,
     &                                      dmA_DF,qmA_DF,omA_DF,hmA_DF,
     &                                      alphaB_DF,AB_DF,CB_DF,EB_DF,
     &                                      U_ind_B
        implicit none
        integer i,j,k,l,m,n
        real*8 term(17)

        U_ind_B=0.d0
        term=0.d0

        do i=1,3,1
        do j=1,3,1
           term(1)=term(1)-T1(i)*T1(j)*qA*qA*alphaB_DF(i,j)/2.d0
        enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           term(2)=term(2)+T1(i)*T2(j,k)*qA*dmA_DF(k)*alphaB_DF(i,j)
           term(3)=term(3)-T1(i)*T2(j,k)*qA*qA*AB_DF(i,j,k)/3.d0
        enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(4)=term(4)-T1(i)*T3(j,k,l)*qA*qmA_DF(k,l)
     &             *alphaB_DF(i,j)/3.d0
           term(5)=term(5)-T2(i,j)*T2(k,l)*dmA_DF(j)*dmA_DF(l)
     &             *alphaB_DF(i,k)/2.d0
           term(6)=term(6)+T1(i)*T3(j,k,l)*qA*dmA_DF(l)
     &             *AB_DF(i,j,k)/3.d0
           term(7)=term(7)+T2(i,j)*T2(k,l)*dmA_DF(j)*qA
     &             *AB_DF(i,k,l)/3.d0
           term(8)=term(8)-T2(i,j)*T2(k,l)*qA*qA*CB_DF(i,j,k,l)/6.d0
           term(9)=term(9)-T1(i)*T3(j,k,l)*qA*qA*EB_DF(i,j,k,l)/15.d0
        enddo;enddo;enddo;enddo

        if(qA>1.d-8.or.qA<-1.d-8)then
           goto 333
        endif

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(10)=term(10)+T2(i,j)*T3(k,l,m)*dmA_DF(j)*qmA_DF(l,m)
     &              *alphaB_DF(i,k)/3.d0
           term(11)=term(11)-T2(i,j)*T3(k,l,m)*dmA_DF(j)*dmA_DF(m)
     &              *AB_DF(i,k,l)/3.d0
        enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           term(12)=term(12)-T2(i,j)*T4(k,l,m,n)*dmA_DF(j)*omA_DF(l,m,n)
     &              *alphaB_DF(i,k)/15.d0
           term(13)=term(13)-T3(i,j,k)*T3(l,m,n)*qmA_DF(j,k)*qmA_DF(m,n)
     &              *alphaB_DF(i,l)/18.d0
           term(14)=term(14)+T2(i,j)*T4(k,l,m,n)*dmA_DF(j)*qmA_DF(m,n)
     &              *AB_DF(i,k,l)/9.d0
           term(15)=term(15)+T3(i,j,k)*T3(l,m,n)*qmA_DF(j,k)*dmA_DF(n)
     &              *AB_DF(i,l,m)/9.d0
           term(16)=term(16)-T3(i,j,k)*T3(l,m,n)*dmA_DF(k)*dmA_DF(n)
     &              *CB_DF(i,j,l,m)/6.d0
           term(17)=term(17)-T2(i,j)*T4(k,l,m,n)*dmA_DF(j)*dmA_DF(n)
     &              *EB_DF(i,k,l,m)/15.d0
        enddo;enddo;enddo;enddo;enddo;enddo

        !term(:)=term(:)*2.1947463137d5
333     U_ind_B=sum(term(1:17))
        !write(*,'(f15.10)')U_ind_B*219474.6
!        write(*,'(6f15.5)')term(1),sum(term(2:3)),sum(term(4:9)),
!     &            sum(term(10:17)),sum(term(18:29)),U_ind_B
        return
        endsubroutine

        subroutine lrdisp
        use variates_for_calculation,only : T2,T3,T4,judisp,
     &                                      UA,UB,
     &                                      alphaA_DF,AA_DF,CA_DF,EA_DF,
     &                                      alphaB_DF,AB_DF,CB_DF,EB_DF,
     &                                      dispdddd_MF,dispdddd_DF,
     &                                      U_disp,U_disp_6,U_disp_7,
     &                                      U_disp_8
        implicit none
        integer i,j,k,l,m,n
        real*8 term(9)

        U_disp=0.d0
        term=0.d0

        if(judisp==0)then

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(1)=term(1)-T2(i,j)*T2(k,l)*alphaA_DF(i,k)*alphaB_DF(j,l)
     &             *UA*UB/(4.d0*(UA+UB))
        enddo;enddo;enddo;enddo

        elseif(judisp==1)then

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           term(1)=term(1)-T2(i,j)*T2(k,l)*dispdddd_DF(i,k,j,l)
        enddo;enddo;enddo;enddo

        endif      

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
           term(2)=term(2)-T2(i,j)*T3(k,l,m)*alphaA_DF(i,k)*AB_DF(j,l,m)
     &             *UA*UB/(6.d0*(UA+UB))
           term(3)=term(3)+T2(i,j)*T3(k,l,m)*AA_DF(i,k,l)*alphaB_DF(j,m)
     &             *UA*UB/(6.d0*(UA+UB))
        enddo;enddo;enddo;enddo;enddo

        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
        do m=1,3,1
        do n=1,3,1
           term(4)=term(4)+T2(i,j)*T4(k,l,m,n)*AA_DF(i,k,l)*AB_DF(j,m,n)
     &             *UA*UB/(18.d0*(UA+UB))
           term(5)=term(5)+T3(i,j,k)*T3(l,m,n)*AA_DF(i,l,m)*AB_DF(n,j,k)
     &             *UA*UB/(18.d0*(UA+UB))
           term(6)=term(6)-T3(i,j,k)*T3(l,m,n)*alphaA_DF(i,l)
     &             *CB_DF(j,k,m,n)*UA*UB/(12.d0*(UA+UB))
           term(7)=term(7)-T3(i,j,k)*T3(l,m,n)*CA_DF(i,j,l,m)
     &             *alphaB_DF(k,n)*UA*UB/(12.d0*(UA+UB))
           term(8)=term(8)-T2(i,j)*T4(k,l,m,n)*alphaA_DF(i,k)
     &             *EB_DF(j,l,m,n)*UA*UB/(30.d0*(UA+UB))
           term(9)=term(9)-T2(i,j)*T4(k,l,m,n)*EA_DF(i,k,l,m)
     &             *alphaB_DF(j,n)*UA*UB/(30.d0*(UA+UB))
        enddo;enddo;enddo;enddo;enddo;enddo

        !term(:)=term(:)*2.1947463137d5
        U_disp_6=term(1)
        U_disp_7=sum(term(2:3))
        U_disp_8=sum(term(4:9))
        U_disp=sum(term(1:9))
        !write(*,'(f15.10)')U_disp*2.1947463137d5
        !write(*,'(4f15.9)')term(4:9)
        return
        endsubroutine


        module elements
        implicit none

        real*8 qmAzz,hmAzzzz !Independent and non-zero when A has D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 dmAz,omAzzz !Independent and non-zero when A has C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 qmAxx,omAxxz,hmAxxxx,hmAxxzz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 dmAx,qmAxz,omAxxx,omAxzz,hmAxxxz,hmAxzzz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 dmAy,qmAxy,qmAyz,omAxxy,omAxyz,omAyzz !Independent and non-zero when A has C_1 symmetry.
        real*8 hmAxxxy,hmAxxyz,hmAxyzz,hmAyzzz !Independent and non-zero when A has C_1 symmetry.

        real*8 alphaAzz,CAzzzz !Independent and non-zero when A has sphere, D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 alphaAxx,CAxyxy,CAxzxz,EAxxzz,EAzzzz !Independent and non-zero when A has D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 AAxxz,AAzzz !Independent and non-zero when A has C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 alphaAyy,AAyyz,AAzxx,CAxxxx,CAxxzz,CAyzyz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 EAxxxx,EAyxxy,EAyyzz,EAzxxz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 alphaAxz,AAxxx,AAxzz,AAyxy,AAzxz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 CAxxxz,CAxyyz,CAxzzz,EAxxxz,EAxzzz,EAyxyz,EAzxxx,EAzxzz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 alphaAxy,alphaAyz !Independent and non-zero when A has C_1 symmetry.
        real*8 AAxxy,AAxyz,AAyxx,AAyxz,AAyzz,AAzxy,AAzyz !Independent and non-zero when A has C_1 symmetry.
        real*8 CAxxxy,CAxxyz,CAxzxy,CAxzyz,CAzzxy,CAzzyz !Independent and non-zero when A has C_1 symmetry.
        real*8 EAxxxy,EAxxyz,EAxyzz,EAyxxx,EAyxxz !Independent and non-zero when A has C_1 symmetry.
        real*8 EAyxzz,EAyzzz,EAzxxy,EAzxyz,EAzyzz !Independent and non-zero when A has C_1 symmetry.

        real*8 qmBzz,hmBzzzz !Independent and non-zero when A has D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 dmBz,omBzzz !Independent and non-zero when A has C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 qmBxx,omBxxz,hmBxxxx,hmBxxzz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 dmBx,qmBxz,omBxxx,omBxzz,hmBxxxz,hmBxzzz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 dmBy,qmBxy,qmByz,omBxxy,omBxyz,omByzz !Independent and non-zero when A has C_1 symmetry.
        real*8 hmBxxxy,hmBxxyz,hmBxyzz,hmByzzz !Independent and non-zero when A has C_1 symmetry.

        real*8 alphaBzz,CBzzzz !Independent and non-zero when B has sphere, D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 alphaBxx,CBxyxy,CBxzxz,EBxxzz,EBzzzz !Independent and non-zero when B has D_inh, C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 ABxxz,ABzzz !Independent and non-zero when A has C_inv, C_2v, C_s, or C_1 symmetry.
        real*8 alphaByy,AByyz,ABzxx,CBxxxx,CBxxzz,CByzyz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 EBxxxx,EByxxy,EByyzz,EBzxxz !Independent and non-zero when A has C_2v, C_s, or C_1 symmetry.
        real*8 alphaBxz,ABxxx,ABxzz,AByxy,ABzxz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 CBxxxz,CBxyyz,CBxzzz,EBxxxz,EBxzzz,EByxyz,EBzxxx,EBzxzz !Independent and non-zero when A has C_s, or C_1 symmetry.
        real*8 alphaBxy,alphaByz !Independent and non-zero when A has C_1 symmetry.
        real*8 ABxxy,ABxyz,AByxx,AByxz,AByzz,ABzxy,ABzyz !Independent and non-zero when A has C_1 symmetry.
        real*8 CBxxxy,CBxxyz,CBxzxy,CBxzyz,CBzzxy,CBzzyz !Independent and non-zero when A has C_1 symmetry.
        real*8 EBxxxy,EBxxyz,EBxyzz,EByxxx,EByxxz !Independent and non-zero when A has C_1 symmetry.
        real*8 EByxzz,EByzzz,EBzxxy,EBzxyz,EBzyzz !Independent and non-zero when A has C_1 symmetry.

        real*8 dispddddxxxx,dispddddxxxy,dispddddxxxz !The independent and non-zero Cartesian components of the dispdddd is simliar with the alphaA*alphaB.
        real*8 dispddddxxyz,dispddddxxyy,dispddddxxzz !However, because the dispdddd is obtained by employing the transition energy and 
        real*8 dispddddxyxx,dispddddxyxy,dispddddxyxz !transition dipole moments, all components of the dispdddd can easily calculated and
        real*8 dispddddxyyz,dispddddxyyy,dispddddxyzz !it is unnecessary to determined which components of dispdddd is independent and non-zero.
        real*8 dispddddxzxx,dispddddxzxy,dispddddxzxz
        real*8 dispddddxzyz,dispddddxzyy,dispddddxzzz
        real*8 dispddddyzxx,dispddddyzxy,dispddddyzxz
        real*8 dispddddyzyz,dispddddyzyy,dispddddyzzz
        real*8 dispddddyyxx,dispddddyyxy,dispddddyyxz
        real*8 dispddddyyyz,dispddddyyyy,dispddddyyzz
        real*8 dispddddzzxx,dispddddzzxy,dispddddzzxz
        real*8 dispddddzzyz,dispddddzzyy,dispddddzzzz
        real*8 e_zero 

        contains

        subroutine get_mult_polar_A
        use variates_for_calculation
        implicit none

        dmA_MF=0.d0;qmA_MF=0.d0;omA_MF=0.d0;hmA_MF=0.d0
        alphaA_MF=0.d0;AA_MF=0.d0;CA_MF=0.d0;EA_MF=0.d0

        call get_dmA
        call get_qmA
        call get_omA
        call get_hmA

        call get_alphaA
        call get_AA
        call get_CA
        call get_EA

        return
        endsubroutine


        subroutine get_mult_polar_B
        use variates_for_calculation
        implicit none

        dmB_MF=0.d0;qmB_MF=0.d0;omB_MF=0.d0;hmB_MF=0.d0
        alphaB_MF=0.d0;AB_MF=0.d0;CB_MF=0.d0;EB_MF=0.d0

        call get_dmB
        call get_qmB
        call get_omB
        call get_hmB

        call get_alphaB
        call get_AB
        call get_CB
        call get_EB

        return
        endsubroutine


        subroutine get_dispdddd
        use variates_for_calculation,only : dispdddd_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        dispdddd_MF=0.d0

        dispdddd_MF(1,1,1,1)=dispddddxxxx
        dispdddd_MF(1,1,1,2)=dispddddxxxy
        dispdddd_MF(1,1,1,2)=dispddddxxxz
        dispdddd_MF(1,1,2,3)=dispddddxxyz
        dispdddd_MF(1,1,2,2)=dispddddxxyy
        dispdddd_MF(1,1,3,3)=dispddddxxzz

        dispdddd_MF(1,2,1,1)=dispddddxyxx
        dispdddd_MF(1,2,1,2)=dispddddxyxy
        dispdddd_MF(1,2,1,2)=dispddddxyxz
        dispdddd_MF(1,2,2,3)=dispddddxyyz
        dispdddd_MF(1,2,2,2)=dispddddxyyy
        dispdddd_MF(1,2,3,3)=dispddddxyzz

        dispdddd_MF(1,3,1,1)=dispddddxzxx
        dispdddd_MF(1,3,1,2)=dispddddxzxy
        dispdddd_MF(1,3,1,2)=dispddddxzxz
        dispdddd_MF(1,3,2,3)=dispddddxzyz
        dispdddd_MF(1,3,2,2)=dispddddxzyy
        dispdddd_MF(1,3,3,3)=dispddddxzzz

        dispdddd_MF(2,3,1,1)=dispddddyzxx
        dispdddd_MF(2,3,1,2)=dispddddyzxy
        dispdddd_MF(2,3,1,2)=dispddddyzxz
        dispdddd_MF(2,3,2,3)=dispddddyzyz
        dispdddd_MF(2,3,2,2)=dispddddyzyy
        dispdddd_MF(2,3,3,3)=dispddddyzzz

        dispdddd_MF(2,2,1,1)=dispddddyyxx
        dispdddd_MF(2,2,1,2)=dispddddyyxy
        dispdddd_MF(2,2,1,2)=dispddddyyxz
        dispdddd_MF(2,2,2,3)=dispddddyyyz
        dispdddd_MF(2,2,2,2)=dispddddyyyy
        dispdddd_MF(2,2,3,3)=dispddddyyzz

        dispdddd_MF(3,3,1,1)=dispddddzzxx
        dispdddd_MF(3,3,1,2)=dispddddzzxy
        dispdddd_MF(3,3,1,2)=dispddddzzxz
        dispdddd_MF(3,3,2,3)=dispddddzzyz
        dispdddd_MF(3,3,2,2)=dispddddzzyy
        dispdddd_MF(3,3,3,3)=dispddddzzzz

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(dispdddd_MF(i,j,k,l)>1.d-8.or.
     &        dispdddd_MF(i,j,k,l)<-1.d-8)then
              tmp=dispdddd_MF(i,j,k,l)
              dispdddd_MF(i,j,l,k)=tmp
              dispdddd_MF(j,i,k,l)=tmp
              dispdddd_MF(j,i,l,k)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_dmA
        use variates_for_calculation, only : dmA_MF
        implicit none

        dmA_MF(1)=dmAx
        dmA_MF(2)=dmAy
        dmA_MF(3)=dmAz

        endsubroutine


        subroutine get_dmB
        use variates_for_calculation, only : dmB_MF
        implicit none

        dmB_MF(1)=dmBx
        dmB_MF(2)=dmBy
        dmB_MF(3)=dmBz

        endsubroutine


        subroutine get_qmA
        use variates_for_calculation,only : symmetryA,qmA_MF
        implicit none

        if(symmetryA==2.or.symmetryA==3)then
           qmA_MF(3,3)=qmAzz
           qmA_MF(2,2)=-0.5d0*qmAzz
           qmA_MF(1,1)=-0.5d0*qmAzz
        elseif(symmetryA==4.or.symmetryA==5.or.symmetryA==6)then
           qmA_MF(1,1)=qmAxx
           qmA_MF(1,2)=qmAxy
           qmA_MF(1,3)=qmAxz
           qmA_MF(2,3)=qmAyz
           qmA_MF(3,3)=qmAzz
           qmA_MF(2,2)=-qmAxx-qmAzz
           qmA_MF(2,1)=qmAxy
           qmA_MF(3,1)=qmAxz
           qmA_MF(3,2)=qmAyz
        endif

        endsubroutine


        subroutine get_qmB
        use variates_for_calculation,only : symmetryB,qmB_MF
        implicit none

        if(symmetryB==2.or.symmetryB==3)then
           qmB_MF(3,3)=qmBzz
           qmB_MF(2,2)=-0.5d0*qmBzz
           qmB_MF(1,1)=-0.5d0*qmBzz
        elseif(symmetryB==4.or.symmetryB==5.or.symmetryB==6)then
           qmB_MF(1,1)=qmBxx
           qmB_MF(1,2)=qmBxy
           qmB_MF(1,3)=qmBxz
           qmB_MF(2,3)=qmByz
           qmB_MF(3,3)=qmBzz
           qmB_MF(2,2)=-qmBxx-qmBzz
           qmB_MF(2,1)=qmBxy
           qmB_MF(3,1)=qmBxz
           qmB_MF(3,2)=qmByz
        endif

        endsubroutine


        subroutine get_omA
        use variates_for_calculation,only : symmetryA,omA_MF
        implicit none

        integer i,j,k
        real*8 tmp

        if(symmetryA==3)then
           omA_MF(3,3,3)=omAzzz
           omA_MF(2,2,3)=-0.5d0*omAzzz
           omA_MF(1,1,3)=-0.5d0*omAzzz
        elseif(symmetryA==4)then
           omA_MF(1,1,3)=omAxxz
           omA_MF(3,3,3)=omAzzz
           omA_MF(2,2,3)=-omAxxz-omAzzz
        elseif(symmetryA==5)then
           omA_MF(1,1,1)=omAxxx
           omA_MF(1,1,3)=omAxxz
           omA_MF(1,3,3)=omAxzz
           omA_MF(3,3,3)=omAzzz
           omA_MF(1,2,2)=-omAxxx-omAxzz
           omA_MF(2,2,3)=-omAxxz-omAzzz
        elseif(symmetryA==6)then
           omA_MF(1,1,1)=omAxxx
           omA_MF(1,1,2)=omAxxy
           omA_MF(1,1,3)=omAxxz
           omA_MF(1,2,3)=omAxyz
           omA_MF(1,3,3)=omAxzz
           omA_MF(2,3,3)=omAyzz
           omA_MF(3,3,3)=omAzzz
           omA_MF(1,2,2)=-omAxxx-omAxzz
           omA_MF(2,2,2)=-omAxxy-omAyzz
           omA_MF(2,2,3)=-omAxxz-omAzzz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           if(omA_MF(i,j,k)>1.d-8.or.omA_MF(i,j,k)<-1.d-8)then
              tmp=omA_MF(i,j,k)
              omA_MF(i,k,j)=tmp
              omA_MF(j,i,k)=tmp
              omA_MF(j,k,i)=tmp
              omA_MF(k,j,i)=tmp
              omA_MF(k,i,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo

        endsubroutine


        subroutine get_omB
        use variates_for_calculation,only : symmetryB,omB_MF
        implicit none

        integer i,j,k
        real*8 tmp

        if(symmetryB==3)then
           omB_MF(3,3,3)=omBzzz
           omB_MF(2,2,3)=-0.5d0*omBzzz
           omB_MF(1,1,3)=-0.5d0*omBzzz
        elseif(symmetryB==4)then
           omB_MF(1,1,3)=omBxxz
           omB_MF(3,3,3)=omBzzz
           omB_MF(2,2,3)=-omBxxz-omBzzz
        elseif(symmetryB==5)then
           omB_MF(1,1,1)=omBxxx
           omB_MF(1,1,3)=omBxxz
           omB_MF(1,3,3)=omBxzz
           omB_MF(3,3,3)=omBzzz
           omB_MF(1,2,2)=-omBxxx-omBxzz
           omB_MF(2,2,3)=-omBxxz-omBzzz
        elseif(symmetryB==6)then
           omB_MF(1,1,1)=omBxxx
           omB_MF(1,1,2)=omBxxy
           omB_MF(1,1,3)=omBxxz
           omB_MF(1,2,3)=omBxyz
           omB_MF(1,3,3)=omBxzz
           omB_MF(2,3,3)=omByzz
           omB_MF(3,3,3)=omBzzz
           omB_MF(1,2,2)=-omBxxx-omBxzz
           omB_MF(2,2,2)=-omBxxy-omByzz
           omB_MF(2,2,3)=-omBxxz-omBzzz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
           if(omB_MF(i,j,k)>1.d-8.or.omB_MF(i,j,k)<-1.d-8)then
              tmp=omB_MF(i,j,k)
              omB_MF(i,k,j)=tmp
              omB_MF(j,i,k)=tmp
              omB_MF(j,k,i)=tmp
              omB_MF(k,j,i)=tmp
              omB_MF(k,i,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo

        endsubroutine


        subroutine get_hmA
        use variates_for_calculation,only : symmetryA,hmA_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryA==2.or.symmetryA==3)then
           hmA_MF(3,3,3,3)=hmAzzzz
           hmA_MF(1,1,3,3)=-0.5d0*hmAzzzz
           hmA_MF(2,2,3,3)=-0.5d0*hmAzzzz
           hmA_MF(1,1,1,1)=3.d0*hmAzzzz/8.d0
           hmA_MF(2,2,2,2)=3.d0*hmAzzzz/8.d0
           hmA_MF(1,1,2,2)=hmAzzzz/8.d0
        elseif(symmetryA==4)then
           hmA_MF(1,1,1,1)=hmAxxxx
           hmA_MF(1,1,3,3)=hmAxxzz
           hmA_MF(3,3,3,3)=hmAzzzz
           hmA_MF(1,1,2,2)=-hmAxxxx-hmAxxzz
           hmA_MF(2,2,3,3)=-hmAxxzz-hmAzzzz
           hmA_MF(2,2,2,2)=hmAxxxx+2.d0*hmAxxzz+hmAzzzz
        elseif(symmetryA==5)then
           hmA_MF(1,1,1,1)=hmAxxxx
           hmA_MF(1,1,3,3)=hmAxxzz
           hmA_MF(3,3,3,3)=hmAzzzz
           hmA_MF(1,1,1,3)=hmAxxxz
           hmA_MF(1,3,3,3)=hmAxzzz
           hmA_MF(1,1,2,2)=-hmAxxxx-hmAxxzz
           hmA_MF(2,2,3,3)=-hmAxxzz-hmAzzzz
           hmA_MF(2,2,2,2)=hmAxxxx+2.d0*hmAxxzz+hmAzzzz
           hmA_MF(1,2,2,3)=-hmAxxxz-hmAxzzz
        elseif(symmetryA==6)then
           hmA_MF(1,1,1,1)=hmAxxxx
           hmA_MF(1,1,3,3)=hmAxxzz
           hmA_MF(3,3,3,3)=hmAzzzz
           hmA_MF(1,1,1,3)=hmAxxxz
           hmA_MF(1,3,3,3)=hmAxzzz
           hmA_MF(2,3,3,3)=hmAyzzz
           hmA_MF(1,2,3,3)=hmAxyzz
           hmA_MF(1,1,2,3)=hmAxxyz
           hmA_MF(1,1,1,2)=hmAxxxy
           hmA_MF(1,1,2,2)=-hmAxxxx-hmAxxzz
           hmA_MF(2,2,3,3)=-hmAxxzz-hmAzzzz
           hmA_MF(2,2,2,2)=hmAxxxx+2.d0*hmAxxzz+hmAzzzz
           hmA_MF(1,2,2,3)=-hmAxxxz-hmAxzzz
           hmA_MF(2,2,2,3)=-hmAxxyz-hmAyzzz
           hmA_MF(1,2,2,2)=-hmAxyzz-hmAxxxy
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(hmA_MF(i,j,k,l)>1.d-8.or.hmA_MF(i,j,k,l)<-1.d-8)then
              tmp=hmA_MF(i,j,k,l)
              hmA_MF(i,j,l,k)=tmp
              hmA_MF(i,k,j,l)=tmp
              hmA_MF(i,k,l,j)=tmp
              hmA_MF(i,l,j,k)=tmp
              hmA_MF(i,l,k,j)=tmp
              hmA_MF(j,i,k,l)=tmp
              hmA_MF(j,i,l,k)=tmp
              hmA_MF(j,k,i,l)=tmp
              hmA_MF(j,k,l,i)=tmp
              hmA_MF(j,l,i,k)=tmp
              hmA_MF(j,l,k,i)=tmp
              hmA_MF(k,i,j,l)=tmp
              hmA_MF(k,i,l,j)=tmp
              hmA_MF(k,j,i,l)=tmp
              hmA_MF(k,j,l,i)=tmp
              hmA_MF(k,l,i,j)=tmp
              hmA_MF(k,l,j,i)=tmp
              hmA_MF(l,i,j,k)=tmp
              hmA_MF(l,i,k,j)=tmp
              hmA_MF(l,j,i,k)=tmp
              hmA_MF(l,j,k,i)=tmp
              hmA_MF(l,k,i,j)=tmp
              hmA_MF(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_hmB
        use variates_for_calculation,only : symmetryB,hmB_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryB==2.or.symmetryB==3)then
           hmB_MF(3,3,3,3)=hmBzzzz
           hmB_MF(1,1,3,3)=-0.5d0*hmBzzzz
           hmB_MF(2,2,3,3)=-0.5d0*hmBzzzz
           hmB_MF(1,1,1,1)=3.d0*hmBzzzz/8.d0
           hmB_MF(2,2,2,2)=3.d0*hmBzzzz/8.d0
           hmB_MF(1,1,2,2)=hmBzzzz/8.d0
        elseif(symmetryB==4)then
           hmB_MF(1,1,1,1)=hmBxxxx
           hmB_MF(1,1,3,3)=hmBxxzz
           hmB_MF(3,3,3,3)=hmBzzzz
           hmB_MF(1,1,2,2)=-hmBxxxx-hmBxxzz
           hmB_MF(2,2,3,3)=-hmBxxzz-hmBzzzz
           hmB_MF(2,2,2,2)=hmBxxxx+2.d0*hmBxxzz+hmBzzzz
        elseif(symmetryB==5)then
           hmB_MF(1,1,1,1)=hmBxxxx
           hmB_MF(1,1,3,3)=hmBxxzz
           hmB_MF(3,3,3,3)=hmBzzzz
           hmB_MF(1,1,1,3)=hmBxxxz
           hmB_MF(1,3,3,3)=hmBxzzz
           hmB_MF(1,1,2,2)=-hmBxxxx-hmBxxzz
           hmB_MF(2,2,3,3)=-hmBxxzz-hmBzzzz
           hmB_MF(2,2,2,2)=hmBxxxx+2.d0*hmBxxzz+hmBzzzz
           hmB_MF(1,2,2,3)=-hmBxxxz-hmBxzzz
        elseif(symmetryB==6)then
           hmB_MF(1,1,1,1)=hmBxxxx
           hmB_MF(1,1,3,3)=hmBxxzz
           hmB_MF(3,3,3,3)=hmBzzzz
           hmB_MF(1,1,1,3)=hmBxxxz
           hmB_MF(1,3,3,3)=hmBxzzz
           hmB_MF(2,3,3,3)=hmByzzz
           hmB_MF(1,2,3,3)=hmBxyzz
           hmB_MF(1,1,2,3)=hmBxxyz
           hmB_MF(1,1,1,2)=hmBxxxy
           hmB_MF(1,1,2,2)=-hmBxxxx-hmBxxzz
           hmB_MF(2,2,3,3)=-hmBxxzz-hmBzzzz
           hmB_MF(2,2,2,2)=hmBxxxx+2.d0*hmBxxzz+hmBzzzz
           hmB_MF(1,2,2,3)=-hmBxxxz-hmBxzzz
           hmB_MF(2,2,2,3)=-hmBxxyz-hmByzzz
           hmB_MF(1,2,2,2)=-hmBxyzz-hmBxxxy
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(hmB_MF(i,j,k,l)>1.d-8.or.hmB_MF(i,j,k,l)<-1.d-8)then
              tmp=hmB_MF(i,j,k,l)
              hmB_MF(i,j,l,k)=tmp
              hmB_MF(i,k,j,l)=tmp
              hmB_MF(i,k,l,j)=tmp
              hmB_MF(i,l,j,k)=tmp
              hmB_MF(i,l,k,j)=tmp
              hmB_MF(j,i,k,l)=tmp
              hmB_MF(j,i,l,k)=tmp
              hmB_MF(j,k,i,l)=tmp
              hmB_MF(j,k,l,i)=tmp
              hmB_MF(j,l,i,k)=tmp
              hmB_MF(j,l,k,i)=tmp
              hmB_MF(k,i,j,l)=tmp
              hmB_MF(k,i,l,j)=tmp
              hmB_MF(k,j,i,l)=tmp
              hmB_MF(k,j,l,i)=tmp
              hmB_MF(k,l,i,j)=tmp
              hmB_MF(k,l,j,i)=tmp
              hmB_MF(l,i,j,k)=tmp
              hmB_MF(l,i,k,j)=tmp
              hmB_MF(l,j,i,k)=tmp
              hmB_MF(l,j,k,i)=tmp
              hmB_MF(l,k,i,j)=tmp
              hmB_MF(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_alphaA
        use variates_for_calculation,only : symmetryA,alphaA_MF
        implicit none

        if(symmetryA==1)then
           alphaA_MF(3,3)=alphaAzz
           alphaA_MF(1,1)=alphaAzz
           alphaA_MF(2,2)=alphaAzz
        elseif(symmetryA==2.or.symmetryA==3)then
           alphaA_MF(1,1)=alphaAxx
           alphaA_MF(3,3)=alphaAzz
           alphaA_MF(2,2)=alphaAxx
        elseif(symmetryA==4.or.symmetryA==5.or.symmetryA==6)then
           alphaA_MF(1,1)=alphaAxx
           alphaA_MF(2,2)=alphaAyy
           alphaA_MF(3,3)=alphaAzz
           alphaA_MF(1,2)=alphaAxy
           alphaA_MF(1,3)=alphaAxz
           alphaA_MF(2,3)=alphaAyz
           alphaA_MF(2,1)=alphaAxy
           alphaA_MF(3,1)=alphaAxz
           alphaA_MF(3,2)=alphaAyz
        endif

        endsubroutine


        subroutine get_alphaB
        use variates_for_calculation,only : symmetryB,alphaB_MF
        implicit none

        if(symmetryB==1)then
           alphaB_MF(3,3)=alphaBzz
           alphaB_MF(1,1)=alphaBzz
           alphaB_MF(2,2)=alphaBzz
        elseif(symmetryB==2.or.symmetryB==3)then
           alphaB_MF(1,1)=alphaBxx
           alphaB_MF(3,3)=alphaBzz
           alphaB_MF(2,2)=alphaBxx
        elseif(symmetryB==4.or.symmetryB==5.or.symmetryB==6)then
           alphaB_MF(1,1)=alphaBxx
           alphaB_MF(2,2)=alphaByy
           alphaB_MF(3,3)=alphaBzz
           alphaB_MF(1,2)=alphaBxy
           alphaB_MF(1,3)=alphaBxz
           alphaB_MF(2,3)=alphaByz
           alphaB_MF(2,1)=alphaBxy
           alphaB_MF(3,1)=alphaBxz
           alphaB_MF(3,2)=alphaByz
        endif

        endsubroutine
           

        subroutine get_AA
        use variates_for_calculation,only :symmetryA,AA_MF
        implicit none

        if(symmetryA==3)then
           AA_MF(1,1,3)=AAxxz
           AA_MF(3,3,3)=AAzzz
           AA_MF(1,3,1)=AAxxz
           AA_MF(3,1,1)=-0.5d0*AAzzz
           AA_MF(3,2,2)=-0.5d0*AAzzz
           AA_MF(2,2,3)=AAxxz
           AA_MF(2,3,2)=AAxxz
        elseif(symmetryA==4)then
           AA_MF(1,1,3)=AAxxz
           AA_MF(3,1,1)=AAzxx
           AA_MF(3,3,3)=AAzzz
           AA_MF(2,2,3)=AAyyz
           AA_MF(3,2,2)=-AAzxx-AAzzz
           AA_MF(1,3,1)=AAxxz
           AA_MF(2,3,2)=AAyyz
        elseif(symmetryA==5)then
           AA_MF(1,1,3)=AAxxz
           AA_MF(3,1,1)=AAzxx
           AA_MF(3,3,3)=AAzzz
           AA_MF(2,2,3)=AAyyz
           AA_MF(1,3,3)=AAxzz
           AA_MF(1,1,1)=AAxxx
           AA_MF(3,1,3)=AAzxz
           AA_MF(2,1,2)=AAyxy
           AA_MF(1,2,2)=-AAxxx-AAxzz
           AA_MF(3,2,2)=-AAzxx-AAzzz
           AA_MF(1,3,1)=AAxxz
           AA_MF(3,3,1)=AAzxz
           AA_MF(2,2,1)=AAyxy
           AA_MF(2,3,2)=AAyyz
        elseif(symmetryA==6)then
           AA_MF(1,1,3)=AAxxz
           AA_MF(3,1,1)=AAzxx
           AA_MF(3,3,3)=AAzzz
           AA_MF(2,2,3)=AAyyz
           AA_MF(1,3,3)=AAxzz
           AA_MF(1,1,1)=AAxxx
           AA_MF(3,1,3)=AAzxz
           AA_MF(2,1,2)=AAyxy

           AA_MF(1,1,2)=AAxxy
           AA_MF(1,2,3)=AAxyz
           AA_MF(2,1,1)=AAyxx
           AA_MF(2,1,3)=AAyxz
           AA_MF(2,3,3)=AAyzz
           AA_MF(3,1,2)=AAzxy
           AA_MF(3,2,3)=AAzyz

           AA_MF(1,2,2)=-AAxxx-AAxzz
           AA_MF(3,2,2)=-AAzxx-AAzzz
           AA_MF(2,2,2)=-AAyxx-AAyzz
           AA_MF(1,3,1)=AAxxz
           AA_MF(3,3,1)=AAzxz
           AA_MF(2,2,1)=AAyxy
           AA_MF(2,3,2)=AAyyz

           AA_MF(1,2,1)=AAxxy
           AA_MF(1,3,2)=AAxyz
           AA_MF(2,3,1)=AAyxz
           AA_MF(3,2,1)=AAzxy
           AA_MF(3,3,2)=AAzyz
        endif

        endsubroutine


        subroutine get_AB
        use variates_for_calculation,only :symmetryB,AB_MF
        implicit none

        if(symmetryB==3)then
           AB_MF(1,1,3)=ABxxz
           AB_MF(3,3,3)=ABzzz
           AB_MF(1,3,1)=ABxxz
           AB_MF(3,1,1)=-0.5d0*ABzzz
           AB_MF(3,2,2)=-0.5d0*ABzzz
           AB_MF(2,2,3)=ABxxz
           AB_MF(2,3,2)=ABxxz
        elseif(symmetryB==4)then
           AB_MF(1,1,3)=ABxxz
           AB_MF(3,1,1)=ABzxx
           AB_MF(3,3,3)=ABzzz
           AB_MF(2,2,3)=AByyz
           AB_MF(3,2,2)=-ABzxx-ABzzz
           AB_MF(1,3,1)=ABxxz
           AB_MF(2,3,2)=AByyz
        elseif(symmetryB==5)then
           AB_MF(1,1,3)=ABxxz
           AB_MF(3,1,1)=ABzxx
           AB_MF(3,3,3)=ABzzz
           AB_MF(2,2,3)=AByyz
           AB_MF(1,3,3)=ABxzz
           AB_MF(1,1,1)=ABxxx
           AB_MF(3,1,3)=ABzxz
           AB_MF(2,1,2)=AByxy
           AB_MF(1,2,2)=-ABxxx-ABxzz
           AB_MF(3,2,2)=-ABzxx-ABzzz
           AB_MF(1,3,1)=ABxxz
           AB_MF(3,3,1)=ABzxz
           AB_MF(2,2,1)=AByxy
           AB_MF(2,3,2)=AByyz
        elseif(symmetryB==6)then
           AB_MF(1,1,3)=ABxxz
           AB_MF(3,1,1)=ABzxx
           AB_MF(3,3,3)=ABzzz
           AB_MF(2,2,3)=AByyz
           AB_MF(1,3,3)=ABxzz
           AB_MF(1,1,1)=ABxxx
           AB_MF(3,1,3)=ABzxz
           AB_MF(2,1,2)=AByxy

           AB_MF(1,1,2)=ABxxy
           AB_MF(1,2,3)=ABxyz
           AB_MF(2,1,1)=AByxx
           AB_MF(2,1,3)=AByxz
           AB_MF(2,3,3)=AByzz
           AB_MF(3,1,2)=ABzxy
           AB_MF(3,2,3)=ABzyz

           AB_MF(1,2,2)=-ABxxx-ABxzz
           AB_MF(3,2,2)=-ABzxx-ABzzz
           AB_MF(2,2,2)=-AByxx-AByzz
           AB_MF(1,3,1)=ABxxz
           AB_MF(3,3,1)=ABzxz
           AB_MF(2,2,1)=AByxy
           AB_MF(2,3,2)=AByyz

           AB_MF(1,2,1)=ABxxy
           AB_MF(1,3,2)=ABxyz
           AB_MF(2,3,1)=AByxz
           AB_MF(3,2,1)=ABzxy
           AB_MF(3,3,2)=ABzyz
        endif

        endsubroutine


        subroutine get_CA
        use variates_for_calculation,only :symmetryA,CA_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryA==1)then
           CA_MF(3,3,3,3)=CAzzzz
           CA_MF(1,1,1,1)=CAzzzz
           CA_MF(2,2,2,2)=CAzzzz
           CA_MF(1,1,3,3)=-0.5d0*CAzzzz
           CA_MF(2,2,3,3)=-0.5d0*CAzzzz
           CA_MF(1,1,2,2)=-0.5d0*CAzzzz
           CA_MF(1,2,1,2)=3.d0*CAzzzz/4.d0
           CA_MF(1,3,1,3)=3.d0*CAzzzz/4.d0
           CA_MF(2,3,2,3)=3.d0*CAzzzz/4.d0
        elseif(symmetryA==2.or.symmetryA==3)then
           CA_MF(3,3,3,3)=CAzzzz
           CA_MF(1,2,1,2)=CAxyxy
           CA_MF(1,3,1,3)=CAxzxz
           CA_MF(1,1,3,3)=-0.5d0*CAzzzz
           CA_MF(2,2,3,3)=-0.5d0*CAzzzz
           CA_MF(1,1,1,1)=CAzzzz/4.d0+CAxyxy
           CA_MF(2,2,2,2)=CAzzzz/4.d0+CAxyxy
           CA_MF(1,1,2,2)=CAzzzz/4.d0-CAxyxy
           CA_MF(2,3,2,3)=CAxzxz
        elseif(symmetryA==4)then
           CA_MF(3,3,3,3)=CAzzzz
           CA_MF(1,2,1,2)=CAxyxy
           CA_MF(1,3,1,3)=CAxzxz
           CA_MF(1,1,1,1)=CAxxxx
           CA_MF(1,1,3,3)=CAxxzz
           CA_MF(2,3,2,3)=CAyzyz
           CA_MF(1,1,2,2)=-CAxxxx-CAxxzz
           CA_MF(2,2,3,3)=-CAxxzz-CAzzzz
           CA_MF(2,2,2,2)=CAxxxx+2.d0*CAxxzz+CAzzzz
        elseif(symmetryA==5)then
           CA_MF(3,3,3,3)=CAzzzz
           CA_MF(1,2,1,2)=CAxyxy
           CA_MF(1,3,1,3)=CAxzxz
           CA_MF(1,1,1,1)=CAxxxx
           CA_MF(1,1,3,3)=CAxxzz
           CA_MF(2,3,2,3)=CAyzyz
           CA_MF(1,1,1,3)=CAxxxz
           CA_MF(1,2,2,3)=CAxyyz
           CA_MF(1,3,3,3)=CAxzzz
           CA_MF(1,1,2,2)=-CAxxxx-CAxxzz
           CA_MF(2,2,3,3)=-CAxxzz-CAzzzz
           CA_MF(2,2,2,2)=CAxxxx+2.d0*CAxxzz+CAzzzz
           CA_MF(2,2,1,3)=-CAxxxz-CAxzzz
        elseif(symmetryA==6)then
           CA_MF(3,3,3,3)=CAzzzz
           CA_MF(1,2,1,2)=CAxyxy
           CA_MF(1,3,1,3)=CAxzxz
           CA_MF(1,1,1,1)=CAxxxx
           CA_MF(1,1,3,3)=CAxxzz
           CA_MF(2,3,2,3)=CAyzyz
           CA_MF(1,1,1,3)=CAxxxz
           CA_MF(1,2,2,3)=CAxyyz
           CA_MF(1,3,3,3)=CAxzzz

           CA_MF(1,1,1,2)=CAxxxy
           CA_MF(1,1,2,3)=CAxxyz
           CA_MF(1,3,1,2)=CAxzxy
           CA_MF(1,3,2,3)=CAxzyz
           CA_MF(3,3,1,2)=CAzzxy
           CA_MF(3,3,2,3)=CAzzyz

           CA_MF(1,1,2,2)=-CAxxxx-CAxxzz
           CA_MF(2,2,3,3)=-CAxxzz-CAzzzz
           CA_MF(2,2,2,2)=CAxxxx+2.d0*CAxxzz+CAzzzz
           CA_MF(2,2,1,3)=-CAxxxz-CAxzzz
           CA_MF(2,2,1,2)=-CAxxxy-CAzzxy
           CA_MF(2,2,2,3)=-CAxxyz-CAzzyz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(CA_MF(i,j,k,l)>1.d-8.or.CA_MF(i,j,k,l)<-1.d-8)then
              tmp=CA_MF(i,j,k,l)
              CA_MF(i,j,l,k)=tmp
              CA_MF(j,i,k,l)=tmp
              CA_MF(j,i,l,k)=tmp
              CA_MF(k,l,i,j)=tmp
              CA_MF(k,l,j,i)=tmp
              CA_MF(l,k,i,j)=tmp
              CA_MF(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_CB
        use variates_for_calculation,only :symmetryB,CB_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryB==1)then
           CB_MF(3,3,3,3)=CBzzzz
           CB_MF(1,1,1,1)=CBzzzz
           CB_MF(2,2,2,2)=CBzzzz
           CB_MF(1,1,3,3)=-0.5d0*CBzzzz
           CB_MF(2,2,3,3)=-0.5d0*CBzzzz
           CB_MF(1,1,2,2)=-0.5d0*CBzzzz
           CB_MF(1,2,1,2)=3.d0*CBzzzz/4.d0
           CB_MF(1,3,1,3)=3.d0*CBzzzz/4.d0
           CB_MF(2,3,2,3)=3.d0*CBzzzz/4.d0
        elseif(symmetryB==2.or.symmetryB==3)then
           CB_MF(3,3,3,3)=CBzzzz
           CB_MF(1,2,1,2)=CBxyxy
           CB_MF(1,3,1,3)=CBxzxz
           CB_MF(1,1,3,3)=-0.5d0*CBzzzz
           CB_MF(2,2,3,3)=-0.5d0*CBzzzz
           CB_MF(1,1,1,1)=CBzzzz/4.d0+CBxyxy
           CB_MF(2,2,2,2)=CBzzzz/4.d0+CBxyxy
           CB_MF(1,1,2,2)=CBzzzz/4.d0-CBxyxy
           CB_MF(2,3,2,3)=CBxzxz
        elseif(symmetryB==4)then
           CB_MF(3,3,3,3)=CBzzzz
           CB_MF(1,2,1,2)=CBxyxy
           CB_MF(1,3,1,3)=CBxzxz
           CB_MF(1,1,1,1)=CBxxxx
           CB_MF(1,1,3,3)=CBxxzz
           CB_MF(2,3,2,3)=CByzyz
           CB_MF(1,1,2,2)=-CBxxxx-CBxxzz
           CB_MF(2,2,3,3)=-CBxxzz-CBzzzz
           CB_MF(2,2,2,2)=CBxxxx+2.d0*CBxxzz+CBzzzz
        elseif(symmetryB==5)then
           CB_MF(3,3,3,3)=CBzzzz
           CB_MF(1,2,1,2)=CBxyxy
           CB_MF(1,3,1,3)=CBxzxz
           CB_MF(1,1,1,1)=CBxxxx
           CB_MF(1,1,3,3)=CBxxzz
           CB_MF(2,3,2,3)=CByzyz
           CB_MF(1,1,1,3)=CBxxxz
           CB_MF(1,2,2,3)=CBxyyz
           CB_MF(1,3,3,3)=CBxzzz
           CB_MF(1,1,2,2)=-CBxxxx-CBxxzz
           CB_MF(2,2,3,3)=-CBxxzz-CBzzzz
           CB_MF(2,2,2,2)=CBxxxx+2.d0*CBxxzz+CBzzzz
           CB_MF(2,2,1,3)=-CBxxxz-CBxzzz
        elseif(symmetryB==6)then
           CB_MF(3,3,3,3)=CBzzzz
           CB_MF(1,2,1,2)=CBxyxy
           CB_MF(1,3,1,3)=CBxzxz
           CB_MF(1,1,1,1)=CBxxxx
           CB_MF(1,1,3,3)=CBxxzz
           CB_MF(2,3,2,3)=CByzyz
           CB_MF(1,1,1,3)=CBxxxz
           CB_MF(1,2,2,3)=CBxyyz
           CB_MF(1,3,3,3)=CBxzzz

           CB_MF(1,1,1,2)=CBxxxy
           CB_MF(1,1,2,3)=CBxxyz
           CB_MF(1,3,1,2)=CBxzxy
           CB_MF(1,3,2,3)=CBxzyz
           CB_MF(3,3,1,2)=CBzzxy
           CB_MF(3,3,2,3)=CBzzyz

           CB_MF(1,1,2,2)=-CBxxxx-CBxxzz
           CB_MF(2,2,3,3)=-CBxxzz-CBzzzz
           CB_MF(2,2,2,2)=CBxxxx+2.d0*CBxxzz+CBzzzz
           CB_MF(2,2,1,3)=-CBxxxz-CBxzzz
           CB_MF(2,2,1,2)=-CBxxxy-CBzzxy
           CB_MF(2,2,2,3)=-CBxxyz-CBzzyz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(CB_MF(i,j,k,l)>1.d-8.or.CB_MF(i,j,k,l)<-1.d-8)then
              tmp=CB_MF(i,j,k,l)
              CB_MF(i,j,l,k)=tmp
              CB_MF(j,i,k,l)=tmp
              CB_MF(j,i,l,k)=tmp
              CB_MF(k,l,i,j)=tmp
              CB_MF(k,l,j,i)=tmp
              CB_MF(l,k,i,j)=tmp
              CB_MF(l,k,j,i)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_EA
        use variates_for_calculation,only :symmetryA,EA_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryA==2.or.symmetryA==3)then
           EA_MF(3,3,3,3)=EAzzzz
           EA_MF(1,1,3,3)=EAxxzz
           EA_MF(3,2,2,3)=-0.5d0*EAzzzz
           EA_MF(3,1,1,3)=-0.5d0*EAzzzz
           EA_MF(1,1,1,1)=-EAxxzz*3.d0/4.d0
           EA_MF(1,1,2,2)=-EAxxzz/4.d0
           EA_MF(2,2,3,3)=EAxxzz
           EA_MF(2,2,2,2)=-EAxxzz*3.d0/4.d0
           EA_MF(2,1,1,2)=-EAxxzz/4.d0
        elseif(symmetryA==4)then
           EA_MF(3,3,3,3)=EAzzzz
           EA_MF(1,1,3,3)=EAxxzz
           EA_MF(1,1,1,1)=EAxxxx
           EA_MF(3,1,1,3)=EAzxxz
           EA_MF(2,1,1,2)=EAyxxy
           EA_MF(2,2,3,3)=EAyyzz
           EA_MF(3,2,2,3)=-EAzzzz-EAzxxz
           EA_MF(1,1,2,2)=-EAxxxx-EAxxzz
           EA_MF(2,2,2,2)=-EAyxxy-EAyyzz
        elseif(symmetryA==5)then
           EA_MF(3,3,3,3)=EAzzzz
           EA_MF(1,1,3,3)=EAxxzz
           EA_MF(1,1,1,1)=EAxxxx
           EA_MF(3,1,1,3)=EAzxxz
           EA_MF(2,1,1,2)=EAyxxy
           EA_MF(2,2,3,3)=EAyyzz

           EA_MF(1,1,1,3)=EAxxxz
           EA_MF(1,3,3,3)=EAxzzz
           EA_MF(2,1,2,3)=EAyxyz
           EA_MF(3,1,1,1)=EAzxxx
           EA_MF(3,1,3,3)=EAzxzz

           EA_MF(3,2,2,3)=-EAzzzz-EAzxxz
           EA_MF(1,1,2,2)=-EAxxxx-EAxxzz
           EA_MF(2,2,2,2)=-EAyxxy-EAyyzz
           EA_MF(1,2,2,3)=-EAxxxz-EAxzzz
           EA_MF(3,1,2,2)=-EAzxxx-EAzxzz
        elseif(symmetryA==6)then
           EA_MF(3,3,3,3)=EAzzzz
           EA_MF(1,1,3,3)=EAxxzz
           EA_MF(1,1,1,1)=EAxxxx
           EA_MF(3,1,1,3)=EAzxxz
           EA_MF(2,1,1,2)=EAyxxy
           EA_MF(2,2,3,3)=EAyyzz

           EA_MF(1,1,1,3)=EAxxxz
           EA_MF(1,3,3,3)=EAxzzz
           EA_MF(2,1,2,3)=EAyxyz
           EA_MF(3,1,1,1)=EAzxxx
           EA_MF(3,1,3,3)=EAzxzz

           EA_MF(1,1,1,2)=EAxxxy
           EA_MF(1,1,2,3)=EAxxyz
           EA_MF(1,2,3,3)=EAxyzz
           EA_MF(2,1,1,1)=EAyxxx
           EA_MF(2,1,1,3)=EAyxxz
           EA_MF(2,1,3,3)=EAyxzz
           EA_MF(2,3,3,3)=EAyzzz
           EA_MF(3,1,1,2)=EAzxxy
           EA_MF(3,1,2,3)=EAzxyz
           EA_MF(3,2,3,3)=EAzyzz

           EA_MF(3,2,2,3)=-EAzzzz-EAzxxz
           EA_MF(1,1,2,2)=-EAxxxx-EAxxzz
           EA_MF(2,2,2,2)=-EAyxxy-EAyyzz
           EA_MF(1,2,2,3)=-EAxxxz-EAxzzz
           EA_MF(3,1,2,2)=-EAzxxx-EAzxzz

           EA_MF(1,2,2,2)=-EAxxxz-EAxyzz
           EA_MF(2,1,2,2)=-EAyxxx-EAyxzz
           EA_MF(2,2,2,3)=-EAyxxz-EAyzzz
           EA_MF(3,2,2,2)=-EAzxxy-EAzyzz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(EA_MF(i,j,k,l)>1.d-8.or.EA_MF(i,j,k,l)<-1.d-8)then
              tmp=EA_MF(i,j,k,l)
              EA_MF(i,j,l,k)=tmp
              EA_MF(i,k,j,l)=tmp
              EA_MF(i,k,l,j)=tmp
              EA_MF(i,l,j,k)=tmp
              EA_MF(i,l,k,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine


        subroutine get_EB
        use variates_for_calculation,only :symmetryB,EB_MF
        implicit none

        integer i,j,k,l
        real*8 tmp

        if(symmetryB==2.or.symmetryB==3)then
           EB_MF(3,3,3,3)=EBzzzz
           EB_MF(1,1,3,3)=EBxxzz
           EB_MF(3,2,2,3)=-0.5d0*EBzzzz
           EB_MF(3,1,1,3)=-0.5d0*EBzzzz
           EB_MF(1,1,1,1)=-EBxxzz*3.d0/4.d0
           EB_MF(1,1,2,2)=-EBxxzz/4.d0
           EB_MF(2,2,3,3)=EBxxzz
           EB_MF(2,2,2,2)=-EBxxzz*3.d0/4.d0
           EB_MF(2,1,1,2)=-EBxxzz/4.d0
        elseif(symmetryB==4)then
           EB_MF(3,3,3,3)=EBzzzz
           EB_MF(1,1,3,3)=EBxxzz
           EB_MF(1,1,1,1)=EBxxxx
           EB_MF(3,1,1,3)=EBzxxz
           EB_MF(2,1,1,2)=EByxxy
           EB_MF(2,2,3,3)=EByyzz
           EB_MF(3,2,2,3)=-EBzzzz-EBzxxz
           EB_MF(1,1,2,2)=-EBxxxx-EBxxzz
           EB_MF(2,2,2,2)=-EByxxy-EByyzz
        elseif(symmetryB==5)then
           EB_MF(3,3,3,3)=EBzzzz
           EB_MF(1,1,3,3)=EBxxzz
           EB_MF(1,1,1,1)=EBxxxx
           EB_MF(3,1,1,3)=EBzxxz
           EB_MF(2,1,1,2)=EByxxy
           EB_MF(2,2,3,3)=EByyzz

           EB_MF(1,1,1,3)=EBxxxz
           EB_MF(1,3,3,3)=EBxzzz
           EB_MF(2,1,2,3)=EByxyz
           EB_MF(3,1,1,1)=EBzxxx
           EB_MF(3,1,3,3)=EBzxzz

           EB_MF(3,2,2,3)=-EBzzzz-EBzxxz
           EB_MF(1,1,2,2)=-EBxxxx-EBxxzz
           EB_MF(2,2,2,2)=-EByxxy-EByyzz
           EB_MF(1,2,2,3)=-EBxxxz-EBxzzz
           EB_MF(3,1,2,2)=-EBzxxx-EBzxzz
        elseif(symmetryB==6)then
           EB_MF(3,3,3,3)=EBzzzz
           EB_MF(1,1,3,3)=EBxxzz
           EB_MF(1,1,1,1)=EBxxxx
           EB_MF(3,1,1,3)=EBzxxz
           EB_MF(2,1,1,2)=EByxxy
           EB_MF(2,2,3,3)=EByyzz

           EB_MF(1,1,1,3)=EBxxxz
           EB_MF(1,3,3,3)=EBxzzz
           EB_MF(2,1,2,3)=EByxyz
           EB_MF(3,1,1,1)=EBzxxx
           EB_MF(3,1,3,3)=EBzxzz

           EB_MF(1,1,1,2)=EBxxxy
           EB_MF(1,1,2,3)=EBxxyz
           EB_MF(1,2,3,3)=EBxyzz
           EB_MF(2,1,1,1)=EByxxx
           EB_MF(2,1,1,3)=EByxxz
           EB_MF(2,1,3,3)=EByxzz
           EB_MF(2,3,3,3)=EByzzz
           EB_MF(3,1,1,2)=EBzxxy
           EB_MF(3,1,2,3)=EBzxyz
           EB_MF(3,2,3,3)=EBzyzz

           EB_MF(3,2,2,3)=-EBzzzz-EBzxxz
           EB_MF(1,1,2,2)=-EBxxxx-EBxxzz
           EB_MF(2,2,2,2)=-EByxxy-EByyzz
           EB_MF(1,2,2,3)=-EBxxxz-EBxzzz
           EB_MF(3,1,2,2)=-EBzxxx-EBzxzz

           EB_MF(1,2,2,2)=-EBxxxz-EBxyzz
           EB_MF(2,1,2,2)=-EByxxx-EByxzz
           EB_MF(2,2,2,3)=-EByxxz-EByzzz
           EB_MF(3,2,2,2)=-EBzxxy-EBzyzz
        endif

        tmp=0.d0
        do i=1,3,1
        do j=1,3,1
        do k=1,3,1
        do l=1,3,1
           if(EB_MF(i,j,k,l)>1.d-8.or.EB_MF(i,j,k,l)<-1.d-8)then
              tmp=EB_MF(i,j,k,l)
              EB_MF(i,j,l,k)=tmp
              EB_MF(i,k,j,l)=tmp
              EB_MF(i,k,l,j)=tmp
              EB_MF(i,l,j,k)=tmp
              EB_MF(i,l,k,j)=tmp
              tmp=0.d0
           endif
        enddo;enddo;enddo;enddo

        endsubroutine

        endmodule

