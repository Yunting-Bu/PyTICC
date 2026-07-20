
        module variates_for_get_mult_polar_disp
        implicit none

        real*8 rA

        contains


        subroutine get_mult_polar_disp
        use elements
        use variates_for_calculation
        implicit none

        call get_elememts
        call get_elements_dispdddd
        call get_mult_polar_A
        call get_mult_polar_B
        call get_dispdddd

        endsubroutine


        subroutine get_elememts
        use parameters_long,only : autoev
        use elements
        use variates_for_calculation,only : UA,UB
        use paraspl,only : ntot
        implicit none
        real*8 elee(ntot)

        call spl1(rA,elee)

        UA=12.070/autoev

        qmAzz=-elee(8)*2.d0  !-0.26268339d0
        alphaAxx=elee(1)     !7.80434628d0
        alphaAzz=elee(2)     !14.89937943d0
        CAzzzz=elee(3)       !21.77978593d0
        CAxyxy=elee(4)       !7.60743086d0
        CAxzxz=elee(5)       !19.05467981d0
        EAzzzz=elee(7)       !20.14325713d0
        EAxxzz=elee(6)       !22.81841308d0
        dispddddxxxx=elee(9) !2.80889422019553d0
        dispddddzzxx=elee(10)!4.67283464344308d0
        e_zero = elee(11)

        UB=13.598d0/autoev
        alphaBzz=4.49293721d0
        CBzzzz=3.97707110d0


        endsubroutine


        subroutine get_elements_dispdddd
        use elements
        implicit none

        dispddddyyxx=dispddddxxxx
        dispddddxxyy=dispddddxxxx
        dispddddyyyy=dispddddxxxx
        dispddddzzyy=dispddddzzxx
        dispddddxxzz=dispddddxxxx
        dispddddyyzz=dispddddxxxx
        dispddddzzzz=dispddddzzxx

        endsubroutine


        endmodule
