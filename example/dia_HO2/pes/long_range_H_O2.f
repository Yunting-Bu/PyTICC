
        include 'long_interactions.f'
        include 'spl.f'
        include 'get_mult_polar_disp.f'

        subroutine get_long(theta,R_jacobi,roo,vlr)
        use variates_for_get_mult_polar_disp
        use variates_for_calculation
        use elements,only : e_zero 
        implicit none
        real*8,parameter :: e0 = -150.66660367d0
        real*8 :: theta,R_jacobi,roo,vlr

        symmetryA=2
        symmetryB=1
        judisp=1

        rA = roo
        betaA = theta
        R = R_jacobi
        call get_mult_polar_disp
        call long_range_interaction

!        vlr = U_int/8065.5d0 + (e_zero-e0)*27.21138d0
!        write(*,*) (e_zero-e0)*27.21138d0
        vlr = U_int 

        end subroutine
