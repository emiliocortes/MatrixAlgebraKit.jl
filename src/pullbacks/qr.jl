qr_rank(R; rank_atol = default_pullback_rank_atol(R)) =
    # @something findlast(>=(rank_atol) ∘ abs, diagview(R)) 0
    # p = count(x -> abs(x) ≥ rank_atol, Rd)
    p = count(x -> abs(x) ≥ rank_atol, diagview(R))

function check_qr_cotangents(
        Q, R, ΔQ, ΔR, p::Int;
        gauge_atol::Real = default_pullback_gauge_atol(ΔQ)
    )
    minmn = min(size(Q, 1), size(R, 2))
    Δgauge = abs(zero(eltype(Q)))
    if !iszerotangent(ΔQ)
        ΔQ₂ = view(ΔQ, :, (p + 1):minmn)
        ΔQ₃ = ΔQ[:, (minmn + 1):size(Q, 2)] # extra columns in the case of qr_full
        Δgauge_Q = norm(ΔQ₂, Inf)
        Q₁ = view(Q, :, 1:p)
        Q₁ᴴΔQ₃ = Q₁' * ΔQ₃
        mul!(ΔQ₃, Q₁, Q₁ᴴΔQ₃, -1, 1)
        Δgauge_Q = max(Δgauge_Q, norm(ΔQ₃, Inf))
        Δgauge = max(Δgauge, Δgauge_Q)
    end
    if !iszerotangent(ΔR)
        ΔR22 = view(ΔR, (p + 1):minmn, (p + 1):size(R, 2))
        Δgauge_R = norm(view(ΔR22, uppertriangularind(ΔR22)), Inf)
        Δgauge_R = max(Δgauge_R, norm(view(ΔR22, diagind(ΔR22)), Inf))
        Δgauge = max(Δgauge, Δgauge_R)
    end
    Δgauge ≤ gauge_atol ||
        @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return nothing
end

"""
    qr_pullback!(
        ΔA, A, QR, ΔQR;
        tol::Real = default_pullback_gaugetol(QR[2]),
        rank_atol::Real = default_pullback_rank_atol(QR[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔQR[1])
    )

Adds the pullback from the QR decomposition of `A` to `ΔA` given the output `QR` and
cotangent `ΔQR` of `qr_compact(A; positive = true)` or `qr_full(A; positive = true)`.

In the case where the rank `r` of the original matrix `A ≈ Q * R` (as determined by
`rank_atol`) is less then the minimum of the number of rows and columns, the cotangents `ΔQ`
and `ΔR`, only the first `r` columns of `Q` and the first `r` rows of `R` are well-defined,
and also the adjoint variables `ΔQ` and `ΔR` should have nonzero values only in the first
`r` columns and rows respectively. If nonzero values in the remaining columns or rows exceed
`gauge_atol`, a warning will be printed.
"""
function qr_pullback!(
        ΔA::AbstractMatrix, A, QR, ΔQR;
        rank_atol::Real = default_pullback_rank_atol(QR[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔQR[1])
    )
    # process
    Q, R = QR
    m = size(Q, 1)
    n = size(R, 2)
    minmn = min(m, n)
    Rd = diagview(R)
    p = qr_rank(R; rank_atol)

    ΔQ, ΔR = ΔQR

    Q₁ = view(Q, :, 1:p)
    R₁₁ = UpperTriangular(view(R, 1:p, 1:p))
    ΔA₁ = view(ΔA, :, 1:p)
    ΔA₂ = view(ΔA, :, (p + 1):n)

    check_qr_cotangents(Q, R, ΔQ, ΔR, p; gauge_atol)

    ΔQ̃ = zero!(similar(Q, (m, p)))
    if !iszerotangent(ΔQ)
        ΔQ₁ = view(ΔQ, :, 1:p)
        copy!(ΔQ̃, ΔQ₁)
        if minmn < size(Q, 2)
            ΔQ₃ = view(ΔQ, :, (minmn + 1):size(ΔQ, 2)) # extra columns in the case of qr_full
            Q₃ = view(Q, :, (minmn + 1):size(Q, 2))
            Q₁ᴴΔQ₃ = Q₁' * ΔQ₃
            ΔQ̃ = mul!(ΔQ̃, Q₃, Q₁ᴴΔQ₃', -1, 1)
        end
    end
    if !iszerotangent(ΔR) && n > p
        R₁₂ = view(R, 1:p, (p + 1):n)
        ΔR₁₂ = view(ΔR, 1:p, (p + 1):n)
        ΔQ̃ = mul!(ΔQ̃, Q₁, ΔR₁₂ * R₁₂', -1, 1)
        # Adding ΔA₂ contribution
        ΔA₂ = mul!(ΔA₂, Q₁, ΔR₁₂, 1, 1)
    end

    # construct M
    M = zero!(similar(R, (p, p)))
    if !iszerotangent(ΔR)
        ΔR₁₁ = UpperTriangular(view(ΔR, 1:p, 1:p))
        M = mul!(M, ΔR₁₁, R₁₁', 1, 1)
    end
    M = mul!(M, Q₁', ΔQ̃, -1, 1)
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end

    # CPU original code
    # rdiv!(M, R₁₁') # R₁₁ is upper triangular
    # rdiv!(ΔQ̃, R₁₁')
    # CUDA friendly alternative
    tmp = similar(M')
    copyto!(tmp, M')
    ldiv!(R₁₁, tmp)
    copyto!(M, tmp')  
    tmp = similar(ΔQ̃')
    copyto!(tmp, ΔQ̃')
    ldiv!(R₁₁, tmp)
    copyto!(ΔQ̃, tmp')
    
    ΔA₁ = mul!(ΔA₁, Q₁, M, +1, 1)
    ΔA₁ .+= ΔQ̃

    return ΔA
end

function check_qr_null_cotangents(N, ΔN; gauge_atol::Real = default_pullback_gauge_atol(ΔN))
    aNᴴΔN = project_antihermitian!(N' * ΔN)
    Δgauge = norm(aNᴴΔN)
    Δgauge ≤ gauge_atol ||
        @warn "`qr_null` cotangent sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
end

"""
    qr_null_pullback!(
        ΔA::AbstractMatrix, A, N, ΔN;
        gauge_atol::Real = default_pullback_gauge_atol(ΔN)
    )

Adds the pullback from the right nullspace of `A` to `ΔA`, given the nullspace basis
`N` and its cotangent `ΔN` of `qr_null(A)`.

See also [`qr_pullback!`](@ref).
"""
function qr_null_pullback!(
        ΔA::AbstractMatrix, A, N, ΔN;
        gauge_atol::Real = default_pullback_gauge_atol(ΔN)
    )
    if !iszerotangent(ΔN) && size(N, 2) > 0
        check_qr_null_cotangents(N, ΔN; gauge_atol)
        Q, R = qr_compact(A; positive = true)
        X = rdiv!(ΔN' * Q, UpperTriangular(R)')
        ΔA = mul!(ΔA, N, X, -1, 1)
    end
    return ΔA
end
