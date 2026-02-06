    .text
    .globl         varint_rvv
    .p2align       1
    .type          varint_rvv,@function
varint_rvv:
    mv             a7,a0
    li             a0,0
    beq            a1,zero,.L_empty
    li             a4,127
    li             t3,-128
    li             t6,16384
.L_loop:
    vsetvli        a5,a1,e8,m1,ta,ma
    vle8.v         v2,0(a7)
    vslide1up.vx   v1,v2,zero
    vmsleu.vx      v3,v2,a4
    vslide1down.vx v4,v2,zero
    vmsleu.vx      v1,v1,a4
    vcpop.m        a3,v3
    vmv1r.v        v0,v1
    vcompress.vm   v3,v2,v0
    vcompress.vm   v5,v4,v0
    beq            a5,a3,.L_all_single_byte
    vslide1down.vx v4,v4,zero
    vand.vx        v9,v3,a4
    vmsgtu.vx      v3,v3,a4
    vslide1down.vx v16,v4,zero
    vcompress.vm   v2,v4,v0
    vmsgtu.vx      v4,v5,a4
    vcompress.vm   v8,v16,v0
    vand.vx        v12,v2,a4
    vmsgtu.vx      v2,v2,a4
    vsetvli        zero,zero,e16,m2,ta,ma
    vzext.vf2      v6,v9
    vsetvli        zero,zero,e8,m1,ta,ma
    vmsgtu.vx      v9,v8,a4
    vand.vx        v5,v5,a4
    vand.vx        v8,v8,a4
    vsetvli        zero,zero,e16,m2,ta,ma
    vmv1r.v        v0,v3
    vzext.vf2      v10,v12
    vmand.mm       v4,v3,v4
    vsetvli        zero,zero,e8,m1,ta,mu
    vwmaccu.vx     v6,t3,v5,v0.t
    vsetvli        zero,a3,e8,m1,ta,ma
    vcpop.m        a6,v3
    vsetvli        zero,a5,e8,m1,ta,ma
    vmand.mm       v0,v4,v2
    vsetvli        zero,a3,e8,m1,ta,ma
    vcpop.m        t1,v4
    vsetvli        zero,a5,e8,m1,ta,mu
    vmand.mm       v9,v0,v9
    vwmaccu.vx     v10,t3,v8,v0.t
    vsetvli        zero,a3,e8,m1,ta,ma
    vcpop.m        t5,v0
    vcpop.m        t4,v9
    add            a6,a6,t1
    vsetvli        zero,a5,e32,m4,ta,ma
    vzext.vf2      v12,v6
    add            a6,a6,a3
    beq            t1,zero,.L_store_advance
    vsetvli        zero,zero,e8,m1,ta,ma
    vmv1r.v        v0,v4
    vslide1down.vx v16,v16,zero
    vsetvli        zero,zero,e16,m2,ta,mu
    vwmaccu.vx     v12,t6,v10,v0.t
    vmv1r.v        v0,v1
    vsetvli        zero,zero,e8,m1,ta,ma
    add            a6,a6,t5
    vcompress.vm   v3,v16,v0
    beq            t4,zero,.L_store_advance
    vand.vx        v3,v3,a4
    vsetvli        zero,zero,e32,m4,ta,mu
    vmv1r.v        v0,v9
    vzext.vf4      v4,v3
    add            a6,a6,t4
    vsll.vi        v4,v4,28
    vadd.vv        v12,v12,v4,v0.t
.L_store_advance:
    vsetvli        zero,a3,e32,m4,ta,ma
    vse32.v        v12,0(a2)
    sub            a1,a1,a6
    add            a7,a7,a6
    sh2add         a2,a3,a2
    add            a0,a0,a3
    bne            a1,zero,.L_loop
    ret
.L_all_single_byte:
    vsetvli        zero,zero,e32,m4,ta,ma
    vzext.vf4      v4,v2
    sub            a1,a1,a5
    add            a7,a7,a5
    vse32.v        v4,0(a2)
    add            a0,a0,a5
    sh2add         a2,a5,a2
    bne            a1,zero,.L_loop
    ret
.L_empty:
    ret
.Lfunc_end0:
    .size          varint_rvv, .Lfunc_end0-varint_rvv


    .globl         varint_rvv_m2
    .p2align       1
    .type          varint_rvv_m2,@function
varint_rvv_m2:
    mv             a7,a0
    li             a0,0
    beq            a1,zero,.Lm2_empty
    li             a3,127
    li             t3,-128
    li             t6,16384
.Lm2_loop:
    vsetvli        a5,a1,e8,m2,ta,ma
    vle8.v         v2,0(a7)
    vslide1up.vx   v10,v2,zero
    vmsleu.vx      v1,v2,a3
    vslide1down.vx v4,v2,zero
    vmsleu.vx      v10,v10,a3
    vcpop.m        a4,v1
    vmv1r.v        v0,v10
    vcompress.vm   v8,v2,v0
    vcompress.vm   v6,v4,v0
    beq            a5,a4,.Lm2_all_single_byte
    vslide1down.vx v4,v4,zero
    vmsgtu.vx      v11,v8,a3
    vmsgtu.vx      v1,v6,a3
    vslide1down.vx v20,v4,zero
    vcompress.vm   v2,v4,v0
    vsetvli        zero,a4,e8,m2,ta,ma
    vand.vx        v8,v8,a3
    vsetvli        zero,a5,e8,m2,ta,ma
    vcompress.vm   v22,v20,v0
    vsetvli        zero,a4,e8,m2,ta,ma
    vand.vx        v12,v2,a3
    vsetvli        zero,a5,e8,m2,ta,ma
    vmsgtu.vx      v2,v2,a3
    vmsgtu.vx      v3,v22,a3
    vsetvli        zero,a4,e8,m2,ta,ma
    vand.vx        v22,v22,a3
    vsetvli        zero,zero,e16,m4,ta,ma
    vzext.vf2      v16,v12
    vsetvli        zero,a5,e8,m2,ta,ma
    vmand.mm       v1,v11,v1
    vsetvli        zero,a4,e16,m4,ta,ma
    vzext.vf2      v12,v8
    vsetvli        zero,a5,e8,m2,ta,ma
    vmand.mm       v0,v1,v2
    vsetvli        zero,a4,e8,m2,ta,ma
    vand.vx        v6,v6,a3
    vsetvli        zero,a5,e8,m2,ta,ma
    vmand.mm       v5,v0,v3
    vsetvli        zero,a4,e8,m2,ta,ma
    vwmaccu.vx     v16,t3,v22,v0.t
    vcpop.m        t1,v1
    vcpop.m        a6,v11
    vcpop.m        t5,v0
    vmv1r.v        v0,v11
    add            a6,a6,t1
    vcpop.m        t4,v5
    vwmaccu.vx     v12,t3,v6,v0.t
    add            a6,a6,a4
    bne            t1,zero,.Lm2_decode_3plus
    vsetvli        zero,a6,e32,m8,ta,ma
    vzext.vf2      v16,v12
    vsetvli        zero,a4,e32,m8,ta,ma
    vse32.v        v16,0(a2)
.Lm2_advance:
    sub            a1,a1,a6
    add            a7,a7,a6
    sh2add         a2,a4,a2
    add            a0,a0,a4
    bne            a1,zero,.Lm2_loop
    ret
.Lm2_all_single_byte:
    vsetvli        zero,zero,e32,m8,ta,ma
    vzext.vf4      v8,v2
    sub            a1,a1,a5
    add            a7,a7,a5
    vse32.v        v8,0(a2)
    add            a0,a0,a5
    sh2add         a2,a5,a2
    bne            a1,zero,.Lm2_loop
    ret
.Lm2_decode_3plus:
    vsetvli        zero,a5,e8,m2,ta,ma
    vslide1down.vx v20,v20,zero
    vsetvli        zero,a4,e32,m8,ta,ma
    vmv1r.v        v0,v10
    vzext.vf2      v24,v12
    vsetvli        zero,a5,e8,m2,ta,ma
    vcompress.vm   v6,v20,v0
    vmv1r.v        v0,v1
    vsetvli        zero,a4,e16,m4,ta,ma
    add            a6,a6,t5
    vwmaccu.vx     v24,t6,v16,v0.t
    beq            t4,zero,.Lm2_store_wide
    vsetvli        zero,zero,e8,m2,ta,ma
    vand.vx        v6,v6,a3
    vsetvli        zero,zero,e32,m8,ta,mu
    vmv1r.v        v0,v5
    vzext.vf4      v16,v6
    add            a6,a6,t4
    vsll.vi        v16,v16,28
    vadd.vv        v24,v24,v16,v0.t
.Lm2_store_wide:
    vse32.v        v24,0(a2)
    j              .Lm2_advance
.Lm2_empty:
    ret
.Lfunc_end1:
    .size          varint_rvv_m2, .Lfunc_end1-varint_rvv_m2
