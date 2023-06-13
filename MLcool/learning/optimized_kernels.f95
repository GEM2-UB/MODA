!
! python -m numpy.f2py -c optimized_kernels.f95 -m optimized_kernels
!

subroutine get_rbf(l, o, n, feat, g, kernel)
implicit none
integer, intent(in)                             :: l, n, o
real(8), dimension(l,o,n), intent(in)           :: feat
real(8), dimension(o), intent(in)               :: g
real(8), dimension(l,l), intent(out)            :: kernel

real(8), dimension(o,n)                         :: samp1, samp2
real(8)                                         :: k_ij
integer                                         :: i, j, k

! l: number of samples
! o: inter, intra
! n: number of features

do i = 1, l, 1
        kernel(i,i) = 1.0D0
end do

do i = 1, l, 1
        samp1(:,:) = feat(i,:,:)
        do j = i+1, l, 1
                samp2(:,:) = feat(j,:,:)
                k_ij = 1.0D0
                do k = 1, o, 1
                        k_ij = k_ij*dexp(-g(k)*sum((samp1(k,:)-samp2(k,:))**2))
                end do
                kernel(i,j) = k_ij
                kernel(j,i) = k_ij

        end do
end do

end subroutine get_rbf


subroutine get_rbf_asym(l1, l2, o, n, feat1, feat2, g, kernel)
implicit none
integer, intent(in)                             :: l1, l2, n, o
real(8), dimension(l1,o,n), intent(in)          :: feat1
real(8), dimension(l2,o,n), intent(in)          :: feat2
real(8), dimension(o), intent(in)               :: g
real(8), dimension(l1,l2), intent(out)          :: kernel

real(8), dimension(o,n)                         :: samp1, samp2
real(8)                                         :: k_ij
integer                                         :: i, j, k

do i = 1, l1, 1
        samp1 = feat1(i,:,:)
        do j = 1, l2, 1
                samp2   = feat2(j,:,:)
                k_ij    = 1.0D0
                do k = 1, o, 1
                        k_ij     = k_ij*dexp(-g(k)*sum((samp1(k,:) - samp2(k,:))**2))
                end do
                kernel(i,j) = k_ij
        end do
end do

end subroutine get_rbf_asym


subroutine get_lap(l, o, n, feat, g, kernel)
implicit none
integer, intent(in)                             :: l, n, o
real(8), dimension(l,o,n), intent(in)           :: feat
real(8), dimension(o), intent(in)               :: g
real(8), dimension(l,l), intent(out)            :: kernel

real(8), dimension(o,n)                         :: samp1, samp2
real(8)                                         :: k_ij
integer                                         :: i, j, k

! l: number of samples
! o: inter, intra
! n: number of features

do i = 1, l, 1
        kernel(i,i) = 1.0D0
end do

do i = 1, l, 1
        samp1(:,:) = feat(i,:,:)
        do j = i+1, l, 1
                samp2(:,:) = feat(j,:,:)
                k_ij = 1.0D0
                do k = 1, o, 1
                        k_ij = k_ij*dexp(-g(k)*sum(abs(samp1(k,:)-samp2(k,:))))
                end do
                kernel(i,j) = k_ij
                kernel(j,i) = k_ij

        end do
end do

end subroutine get_lap


subroutine get_lap_asym(l1, l2, o, n, feat1, feat2, g, kernel)
implicit none
integer, intent(in)                             :: l1, l2, n, o
real(8), dimension(l1,o,n), intent(in)          :: feat1
real(8), dimension(l2,o,n), intent(in)          :: feat2
real(8), dimension(o), intent(in)               :: g
real(8), dimension(l1,l2), intent(out)          :: kernel

real(8), dimension(o,n)                         :: samp1, samp2
real(8)                                         :: k_ij
integer                                         :: i, j, k

do i = 1, l1, 1
        samp1 = feat1(i,:,:)
        do j = 1, l2, 1
                samp2   = feat2(j,:,:)
                k_ij    = 1.0D0
                do k = 1, o, 1
                        k_ij     = k_ij*dexp(-g(k)*sum(abs(samp1(k,:) - samp2(k,:))))
                end do
                kernel(i,j) = k_ij
        end do
end do

end subroutine get_lap_asym


subroutine get_normalization_attributes(l, o, n, feat, axis_max)
implicit none
integer, intent(in)                             :: l, o, n
real(8), dimension(l, o, n), intent(in)         :: feat
real(8), dimension(o), intent(out)              :: axis_max

real(8)                                         :: mx
integer                                         :: i, j

axis_max = -1.0E10
do i = 1, l, 1
        do j = 1, o, 1
                mx          = maxval(abs(feat(i,j,:)))
                axis_max(j) = max(axis_max(j), mx)
        end do
end do

end subroutine get_normalization_attributes


subroutine get_normalization(l, o, n, feat, axis_max, norm_feat)
implicit none
integer, intent(in)                             :: l, o, n
real(8), dimension(l, o, n), intent(in)         :: feat
real(8), dimension(l, o, n), intent(out)        :: norm_feat
real(8), dimension(o), intent(in)               :: axis_max

integer                                         :: i, j

do i = 1, l, 1
        do j = 1, o, 1
                norm_feat(i,j,:) = feat(i,j,:)/axis_max(j)
        end do
end do

end subroutine get_normalization
