    .text
    .globl         varint_rvv
    .p2align       1
    .type          varint_rvv,@function
varint_rvv:
    mv             a7, a0
    li             a0, 0
    beq            a1, zero, .L7
    li             a3, 127
    li             t3, -128
    li             t0, 16384
.L6:
    vsetvli        a5, a1, e8, m1, ta, ma
    vle8.v         v5, 0(a7)
    vslide1up.vx   v2, v5, zero
    vslide1down.vx v1, v5, zero
    vmsleu.vx      v3, v5, a3
    vmsleu.vx      v2, v2, a3
    vslide1down.vx v4, v1, zero
    vcpop.m        a4, v3
    beq            a5, a4, .fastpath
    vmv1r.v        v0, v2
    vslide1down.vx v16, v4, zero
    vcompress.vm   v3, v5, v0
    vcompress.vm   v6, v1, v0
    vcompress.vm   v1, v4, v0
    vsetvli        zero, a4, e8, m1, ta, ma
    vand.vx        v10, v3, a3
    vsetvli        zero, a5, e8, m1, ta, ma
    vmsgtu.vx      v3, v3, a3
    vcompress.vm   v8, v16, v0
    vmsgtu.vx      v4, v6, a3
    vsetvli        zero, a4, e8, m1, ta, ma
    vand.vx        v9, v6, a3
    vand.vx        v12, v1, a3
    vsetvli        zero, zero, e16, m2, ta, ma
    vzext.vf2      v6, v10
    vsetvli        zero, a5, e8, m1, ta, ma
    vmv1r.v        v0, v3
    vmand.mm       v4, v3, v4
    vmsgtu.vx      v1, v1, a3
    vsetvli        zero, a4, e8, m1, ta, mu
    vwmaccu.vx     v6, t3, v9, v0.t
    vsetvli        zero, a5, e8, m1, ta, ma
    vmsgtu.vx      v9, v8, a3
    vsetvli        zero, a4, e8, m1, ta, ma
    vand.vx        v8, v8, a3
    vsetvli        zero, zero, e16, m2, ta, ma
    vzext.vf2      v10, v12
    vsetvli        zero, a5, e8, m1, ta, ma
    vmand.mm       v0, v4, v1
    vsetvli        zero, a4, e8, m1, ta, ma
    vcpop.m        t6, v3
    vsetvli        zero, a5, e8, m1, ta, ma
    vmand.mm       v9, v0, v9
    vsetvli        zero, a4, e8, m1, ta, mu
    vwmaccu.vx     v10, t3, v8, v0.t
    vcpop.m        t5, v4
    vsetvli        zero, zero, e32, m4, ta, ma
    vzext.vf2      v12, v6
    vcpop.m        t4, v0
    vcpop.m        t1, v9
    vsetvli        zero, a5, e8, m1, ta, ma
    vmv1r.v        v0, v2
    vslide1down.vx v16, v16, zero
    add            a6, a4, t1
    add            a6, a6, t6
    vcompress.vm   v3, v16, v0
    vsetvli        zero, a4, e8, m1, ta, ma
    vmv1r.v        v0, v4
    vand.vx        v3, v3, a3
    add            a5, a6, t5
    vsetvli        zero, zero, e16, m2, ta, mu
    vwmaccu.vx     v12, t0, v10, v0.t
    add            a5, a5, t4
    beq            t1, zero, .skipFiveBytes
    vsetvli        zero, zero, e32, m4, ta, mu
    vzext.vf4      v4, v3
    vmv1r.v        v0, v9
    vsll.vi        v4, v4, 28
    vadd.vv        v12, v12, v4, v0.t
.skipFiveBytes:
    vse32.v        v12, 0(a2)
    sub            a1, a1, a5
    add            a7, a7, a5
    sh2add         a2, a4, a2
    add            a0, a0, a4
    bne            a1, zero, .L6
    ret
.fastpath:
    vsetvli        zero, a5, e32, m4, ta, ma
    vzext.vf4      v8, v5
    sub            a1, a1, a5
    add            a7, a7, a5
    vse32.v        v8, 0(a2)
    add            a0, a0, a5
    sh2add         a2, a5, a2
    bne            a1, zero, .L6
    ret
.L7:
    ret
.Lfunc_end0:
    .size          varint_rvv, .Lfunc_end0-varint_rvv
