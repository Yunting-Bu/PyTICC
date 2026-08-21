      PROGRAM PESRCCSDTHENO
C PROGRAM TO GENERATE PES GRID FOR CONTOURS 
      IMPLICIT DOUBLE PRECISION  ( A-H, O-Z )
      DIMENSION R(121),ANG(91),X(401),Y(401)
      DO I=1,121
      R(I)=4.0d0+0.05d0*DFLOAT(I-1)
      ENDDO
      DO I=1,401
      X(I)=-10.d0+0.05d0*DFLOAT(I-1)
      Y(I)=X(I)
      ENDDO
      DO J=1,91
      ANG(J)=2.D0*DFLOAT(J-1)
      ENDDO
      DO I=1,121
      DO J=1,91
      COSS=DCOS(ANG(J)*3.141592653d0/180.d0)
      COSS2=DCOS((180.d0-ANG(J))*3.141592653d0/180.d0)
C      RR=DSQRT(X(I)**2+Y(J)**2)
C      COSS=X(I)/RR
C      IF(DABS(X(I)).LT.2.5D0.AND.DABS(Y(J)).LT.2.5D0) THEN
C      ELSE
c      WRITE(6,*) X(I),Y(J),VSUM(RR,COSS)+VDIF(RR,COSS)
c     1 ,VSUM(RR,COSS)-VDIF(RR,COSS),VSUM(RR,COSS)
      WRITE(6,*) R(I),ANG(J),0.5D0*(VSUM(R(I),COSS)+VSUM(R(I),COSS2)),
     1    0.5D0*(VDIF(R(I),COSS)+VDIF(R(I),COSS2))
C      ENDIF
      ENDDO
      ENDDO
      END
*Decks for PES He-NO

      FUNCTION VDIF(R,C)
*FUNCTION  FOR HENO Vdif=0.5(A"-A')
*RCCSD(T) / AVTZ+332
*INPUT in bohrs and cosine(theta)
*OUTPUT in wavenumbers
*J. Klos,S. Cybulski,G. Chalasinski at al.
*Reference:
*J. Chem. Phys. 112,2195 (2000) 
      IMPLICIT REAL*8  (A-H,O-Z)
      IF (DABS(C).EQ.1.D0) THEN
      VDIF=0.D0
      ELSE
      VDIF= 0.5d0*(POT1(R,C)-POT2(R,C))*
     1  1000.D0/4.5563357d0
      ENDIF
      RETURN
      END
      FUNCTION VSUM(R,C)
*FUNCTION  FOR HENO Vsum=0.5(A"+A')
*RCCSD(T)/ AVTZ+332
*INPUT in bohrs and cosine(theta)
*OUTPUT in wavenumbers
*J. Klos,S. Cybulski,G. Chalasinski et al.
*Reference:
*J. Chem. Phys. 112,2195 (2000)

      IMPLICIT REAL*8  (A-H,O-Z)
      VSUM= 0.5d0*(POT1(R,C)+POT2(R,C))*
     1  1000.D0/4.5563357d0
      RETURN
      END
      



      FUNCTION POT1(R,C)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8 A(6)
      PARAMETER (ME = 6, ML = 6,
     *  MT = 2 * ME + 4 * ML + 4 )
      PARAMETER (ONE = 1.0D0, THREE = 3.0D0,
     * F15 = 15.0D0,
     *  ZERO = 0.0D0, FIVE = 5.0D0, THIRTY = 30.0D0,
     *  SEVNTY = 70.0D0, HALF = 0.5D0, T35 = 35.0D0,
     *   S63 = 63.0D0, THRESH = 1.0D-8) 
      REAL*8    X(MT)
*
*     Potential energy function for 
*      a rare gas atom - asymmetric
*     linear molecule complexes.
*
*     On input:
*     =========
*R - intermolecular distance (in au)
*C - cos( theta ), where theta is the angle between
*     the bond axis
*    of a diatomic and a vector from Rg to the
*    center of mass
*    of a diatomic
*
*  On return:
*  ==========
*  POTFUN - interaction energy in mE_h (milihartrees)
       data X/0.93729727E+01,-0.34394679,
     *0.12075059E+01,
     *   0.87512578E-01,
     *  0.24040713E+00,0.14739815E+00,-0.20236282E+01,
     *  -0.44819367E-01,
     * -0.18935935E-01, 0.43253788E-04,-0.44905825E-01,
     *  -0.42649430E-01,
     * -0.25182060E+01, 0.21820896E+01,-0.35647710E+01,
     *  0.56402415E+00,
     *  0.60600803E+01, 0.24719691E+00,0.38623219E+01,
     *   0.26295718E+00,
     *  0.23270757E+01, -0.10360823E+01,-0.39459956E+01,
     *  -0.67839402E+00,
     * -0.53872460E+00,-0.11447268E+00,-0.37394241E+00,
     *   0.24835212E+00,
     *  0.78682216E+00,0.24061793E+00,0.90072824E-02,
     *   0.39021151E-02,
     * 0.18131326E-01,-0.17611395E-01,-0.49279888E-01,
     *   -0.19927465E-01,
     * -0.12539203E+05,-0.47580271E+04,0.15366833E+05,
     *  0.51647119E+04/

*A" state
*INSERT THE PARAMETERS HERE
*
*
*     Start by evaluating the required Legendre polynomials
*
      T = -C
      T2 = T * T
      T3 = T2 * T
      T4 = T3 * T
      T5 = T4 * T
      T6 = T5 * T
      A( 1 ) = ONE
      A( 2 ) = T
      A( 3 ) = HALF * ( THREE * T2 - ONE )
      A( 4 ) = HALF * ( FIVE * T3 - THREE * T )
      A( 5 ) = 0.125D0 * ( T35 * T4 - THIRTY * T2 + THREE )
      A( 6 ) = 0.125D0 * ( S63 * T5 - SEVNTY * T3 + F15 * T )
*
*     VSH( R, THETA ) part:
*
      D = ZERO
      B = ZERO
      DO 100 I = 1, ME
         D = D + X( I ) * A( I )
         B = B + X( I + ME ) * A( I )
 100  CONTINUE
      NE = 2 * ME
      R2 = R * R
      R3 = R2 * R
      G = ZERO
      DO 200 I = 1, ML
         IE = I + NE
         G = G + ( X( IE ) + X( IE + ML ) * R + X( IE + 2 * ML )
     *         * R2 + X( IE + 3 * ML ) * R3 ) * A( I )
 200  CONTINUE
      VSH = G * EXP( D + B * R )
*
*     VAS( R, THETA ) part:
*
      BR = B * R
      S = ONE
      T = ONE
      DBR = DABS( BR )
      DO 300 K = 1, 6
         T = T * DBR / DFLOAT( K )
         S = S + T
 300  CONTINUE
      F6 = ONE - DEXP( BR ) * S
      T = T * DBR / DFLOAT( 7 )
      S = S + T
      F7 = ONE - DEXP( BR ) * S
*
*     If F6 or f7 are smaller than 1.0D-8,
*       they should be recalculated
*     and the segment below will do the trick.
*
      IF( DABS( F6 ) .LT. THRESH ) THEN
         F6 = ZERO
         DO 400 I = 7, 1000
            T = T * BR / DFLOAT( I )
            F6 = F6 + T
            IF( ( T / F6 ) .LT. THRESH ) GO TO 500
 400     CONTINUE
         WRITE( 6, * ) 'No convergence for F6.'
         STOP
 500     CONTINUE
         F6 = F6 * DEXP( BR )
      END IF
*
      IF( DABS( F7 ) .LT. THRESH ) THEN
         F7 = ZERO
         DO 600 I = 8, 1000
            T = T * BR / DFLOAT( I )
            F7 = F7 + T
            IF( ( T / F7 ) .LT. THRESH ) GO TO 700
 600     CONTINUE
         WRITE( 6, * ) 'No convergence for F7.'
 700     CONTINUE
         F7 = F7 * DEXP( BR )
      END IF
*
      R6 = R3 * R3
      R7 = R6 * R
      NE = NE + 4 * ML
      VAS = F6 * ( X( 1 + NE ) + X( 2 + NE ) * A( 3 ) ) / R6
     *    + F7 * ( X( 3 + NE )
     *  * A( 2 ) + X( 4 + NE ) * A( 4 ) ) / R7
      POT1 = VSH + VAS
      RETURN
      END
*Deck POT2 FOR A'
      FUNCTION POT2(R,C)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8 A(6)
      PARAMETER (ME = 6, ML = 6,
     *  MT = 2 * ME + 4 * ML + 4 )
      PARAMETER (ONE = 1.0D0, THREE = 3.0D0,
     * F15 = 15.0D0,
     *  ZERO = 0.0D0, FIVE = 5.0D0, THIRTY = 30.0D0,
     *  SEVNTY = 70.0D0, HALF = 0.5D0, T35 = 35.0D0,
     *   S63 = 63.0D0, THRESH = 1.0D-8) 
      REAL*8    X(MT)
*
*     Potential energy function for 
*      a rare gas atom - asymmetric ( case A' HeNO RCCSD(T))
*     linear molecule complexes.
*
*     On input:
*     =========
*R - intermolecular distance (in au)
*C - cos( theta ), where theta is the angle between
*     the bond axis
*    of a diatomic and a vector from Rg to the
*    center of mass
*    of a diatomic
*
*  On return:
*  ==========
*  POTFUN - interaction energy in mE_h (milihartrees)
      data X/0.89614790D+01,-0.66184380D+00,
     *  0.78007803D+00,-0.60974193D-02,
     * -0.35808998D-01, -0.11214555D-01,
     * -0.20242288D+01, -0.34243587D-01,
     * -0.33588331D-01, -0.37086659D-02,
     *  0.77114573D-02,  0.24071004D-02,
     * -0.24565438D+00,  0.32383547D+01,
     * -0.18898698D+01,  0.13834898D+01,
     *  0.23578296D+01,  0.14503735D+01,
     *  0.43039404D+01,  0.22754183D+01,
     *  0.47018602D+01,  0.24381747D+00,
     * -0.25353828D+00, -0.49050007D+00,
     * -0.44107679D+00, -0.38335519D+00,
     * -0.55140365D+00,  0.42936877D-01,
     * -0.19769691D+00,  0.19501492D-01,
     * -0.74599920D-02, -0.40805756D-02,
     *  0.23613677D-07, -0.56473632D-02,
     *  0.11274139D-01, -0.33655160D-02,
     * -0.13246868D+05, -0.41907209D+04,
     *  0.20463763D+05,  0.58529744D+03/    
*A1 state HeNO
*INSERT THE PARAMETERS HERE
*
*
*     Start by evaluating the required Legendre polynomials
*
      T = -C
      T2 = T * T
      T3 = T2 * T
      T4 = T3 * T
      T5 = T4 * T
      T6 = T5 * T
      A( 1 ) = ONE
      A( 2 ) = T
      A( 3 ) = HALF * ( THREE * T2 - ONE )
      A( 4 ) = HALF * ( FIVE * T3 - THREE * T )
      A( 5 ) = 0.125D0 * ( T35 * T4 - THIRTY * T2 + THREE )
      A( 6 ) = 0.125D0 * ( S63 * T5 - SEVNTY * T3 + F15 * T )
*
*     VSH( R, THETA ) part:
*
      D = ZERO
      B = ZERO
      DO 100 I = 1, ME
         D = D + X( I ) * A( I )
         B = B + X( I + ME ) * A( I )
 100  CONTINUE
      NE = 2 * ME
      R2 = R * R
      R3 = R2 * R
      G = ZERO
      DO 200 I = 1, ML
         IE = I + NE
         G = G + ( X( IE ) + X( IE + ML ) * R + X( IE + 2 * ML )
     *         * R2 + X( IE + 3 * ML ) * R3 ) * A( I )
 200  CONTINUE
      VSH = G * EXP( D + B * R )
*
*     VAS( R, THETA ) part:
*
      BR = B * R
      S = ONE
      T = ONE
      DBR = DABS( BR )
      DO 300 K = 1, 6
         T = T * DBR / DFLOAT( K )
         S = S + T
 300  CONTINUE
      F6 = ONE - DEXP( BR ) * S
      T = T * DBR / DFLOAT( 7 )
      S = S + T
      F7 = ONE - DEXP( BR ) * S
*
*     If F6 or f7 are smaller than 1.0D-8,
*       they should be recalculated
*     and the segment below will do the trick.
*
      IF( DABS( F6 ) .LT. THRESH ) THEN
         F6 = ZERO
         DO 400 I = 7, 1000
            T = T * BR / DFLOAT( I )
            F6 = F6 + T
            IF( ( T / F6 ) .LT. THRESH ) GO TO 500
 400     CONTINUE
         WRITE( 6, * ) 'No convergence for F6.'
         STOP
 500     CONTINUE
         F6 = F6 * DEXP( BR )
      END IF
*
      IF( DABS( F7 ) .LT. THRESH ) THEN
         F7 = ZERO
         DO 600 I = 8, 1000
            T = T * BR / DFLOAT( I )
            F7 = F7 + T
            IF( ( T / F7 ) .LT. THRESH ) GO TO 700
 600     CONTINUE
         WRITE( 6, * ) 'No convergence for F7.'
 700     CONTINUE
         F7 = F7 * DEXP( BR )
      END IF
*
      R6 = R3 * R3
      R7 = R6 * R
      NE = NE + 4 * ML
      VAS = F6 * ( X( 1 + NE ) + X( 2 + NE ) * A( 3 ) ) / R6
     *    + F7 * ( X( 3 + NE )
     *  * A( 2 ) + X( 4 + NE ) * A( 4 ) ) / R7
      POT2 = VSH + VAS
      RETURN
      END
