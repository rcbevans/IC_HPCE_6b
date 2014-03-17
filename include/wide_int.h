#ifndef wide_int_h
#define wide_int_h

#include <stdint.h>
#include <assert.h>

#include <stdio.h>
#include <math.h>

typedef __uint128_t u128b;
struct uint128b
{
    uint64_t lo;
    uint64_t mid;
};

/*! Simply library for maintaining large positive integers as an array
    of 32-bit limbs */

void wide_x_init(uint32_t *x, const uint32_t index, const uint64_t roundId, const uint64_t roundSalt, const uint64_t chainHash)
{
    x[0] = index;
    x[1] = 0;

    uint64_t acc = (roundId & 0xFFFFFFFFULL);
    x[2] = uint32_t(acc & 0xFFFFFFFFULL);
    uint64_t carry = acc >> 32;

    acc = (roundId & 0xFFFFFFFFULL) + carry;
    x[3] = uint32_t(acc & 0xFFFFFFFFULL);

    acc = (roundSalt & 0xFFFFFFFFULL);
    x[4] = uint32_t(acc & 0xFFFFFFFFULL);
    carry = acc >> 32;

    acc = (roundSalt & 0xFFFFFFFFULL) + carry;
    x[5] = uint32_t(acc & 0xFFFFFFFFULL);

    acc = (chainHash & 0xFFFFFFFFULL);
    x[6] = uint32_t(acc & 0xFFFFFFFFULL);
    carry = acc >> 32;

    acc = (chainHash & 0xFFFFFFFFULL) + carry;
    x[7] = uint32_t(acc & 0xFFFFFFFFULL);
}

/*! Compare two integers as numbers:
        a<b :  -1
        a==b :  0
        a>b : +1
*/
int wide_compare(unsigned n, const uint32_t *a, const uint32_t *b)
{
    if (a == b)
        return 0;

    for (int i = n - 1; i >= 0; i--)
    {
        if (a[i] < b[i])
            return -1;
        if (a[i] > b[i])
            return +1;
    }
    return 0;
}

/*! Copy a source number to a destination */
void wide_copy(unsigned n, uint32_t *res, const uint32_t *a)
{
    for (unsigned i = 0; i < n; i++)
    {
        res[i] = a[i];
    }
}

/*! Set entire number to zero */
void wide_zero(unsigned n, uint32_t *res)
{
    for (unsigned i = 0; i < n; i++)
    {
        res[i] = 0;
    }
}

/*! Set entire number to zero */
void wide_ones(unsigned n, uint32_t *res)
{
    for (unsigned i = 0; i < n; i++)
    {
        res[i] = 0xFFFFFFFFul;
    }
}

/*! Add together two n-limb numbers, returning the carry limb.
    \note the output can also be one of the inputs
*/
void wide_xor(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
    for (unsigned i = 0; i < n; i++)
    {
        res[i] = a[i] ^ b[i];
    }
}

/*! Add together two n-limb numbers, returning the carry limb.
    \note the output can also be one of the inputs
*/
uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
    uint64_t carry = 0;
    for (unsigned i = 0; i < n; i++)
    {
        uint64_t tmp = uint64_t(a[i]) + b[i] + carry;
        res[i] = uint32_t(tmp & 0xFFFFFFFFULL);
        carry = tmp >> 32;
    }
    return carry;
}


/*! Add a single limb to an n-limb number, returning the carry limb
\note the output can also be the input
*/
uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint32_t b)
{
    uint64_t carry = b;
    for (unsigned i = 0; i < n; i++)
    {
        uint64_t tmp = a[i] + carry;
        res[i] = uint32_t(tmp & 0xFFFFFFFFULL);
        carry = tmp >> 32;
    }
    return carry;
}

uint32_t fast_wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint32_t b)
{
    uint64_t carry = b;
    for (unsigned i = 0; i < n; i++)
    {
        uint64_t tmp = a[i] + carry;
        res[i] = uint32_t(tmp & 0xFFFFFFFFULL);
        carry = tmp >> 32;
        if (carry == 0)
        {
            for (unsigned j = i + 1; j < n; j++)
            {
                res[j] = a[j];
            }
            break;
        }
    }
    return carry;
}

uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint64_t b)
{
    assert(n >= 2);
    uint64_t acc = a[0] + (b & 0xFFFFFFFFULL);
    res[0] = uint32_t(acc & 0xFFFFFFFFULL);
    uint64_t carry = acc >> 32;

    acc = a[1] + (b & 0xFFFFFFFFULL) + carry;
    res[1] = uint32_t(acc & 0xFFFFFFFFULL);
    carry = acc >> 32;

    for (unsigned i = 2; i < n; i++)
    {
        uint64_t tmp = a[i] + carry;
        res[i] = uint32_t(tmp & 0xFFFFFFFFULL);
        carry = tmp >> 32;
    }
    return carry;
}

uint32_t fast_wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint64_t b)
{
    assert(n >= 2);
    uint64_t acc = a[0] + (b & 0xFFFFFFFFULL);
    res[0] = uint32_t(acc & 0xFFFFFFFFULL);
    uint64_t carry = acc >> 32;

    acc = a[1] + (b & 0xFFFFFFFFULL) + carry;
    res[1] = uint32_t(acc & 0xFFFFFFFFULL);
    carry = acc >> 32;

    for (unsigned i = 2; i < n; i++)
    {
        uint64_t tmp = a[i] + carry;
        res[i] = uint32_t(tmp & 0xFFFFFFFFULL);
        carry = tmp >> 32;
        if (carry == 0)
        {
            for (unsigned j = i + 1; j < n; j++)
            {
                res[j] = a[j];
            }
            break;
        }
    }
    return carry;
}

// if (!carry)
// {
// for(unsigned j = i+1; j < n; j++)
// {
//  res[j] = a[i];
// }
// break;
// }

/*! Multiply two n-limb numbers to produce a 2n-limb result
    \note All the integers must be distinct, the output cannot overlap the input */
void wide_mul(unsigned n, uint32_t *res_hi, uint32_t *res_lo, const uint32_t *a, const uint32_t *b)
{
    assert(res_hi != a && res_hi != b);
    assert(res_lo != a && res_lo != b);

    uint64_t carry = 0, acc = 0;
    for (unsigned i = 0; i < n; i++)
    {
        for (unsigned j = 0; j <= i; j++)
        {
            assert( (j + (i - j)) == i );
            uint64_t tmp = uint64_t(a[j]) * b[i - j];
            acc += tmp;
            if (acc < tmp)
                carry++;
        }
        res_lo[i] = uint32_t(acc & 0xFFFFFFFFull);
        acc = (carry << 32) | (acc >> 32);
        carry = carry >> 32;
    }

    for (unsigned i = 1; i < n; i++)
    {
        for (unsigned j = i; j < n; j++)
        {
            uint64_t tmp = uint64_t(a[j]) * b[n - j + i - 1];
            acc += tmp;
            if (acc < tmp)
                carry++;
        }
        res_hi[i - 1] = uint32_t(acc & 0xFFFFFFFFull);
        acc = (carry << 32) | (acc >> 32);
        carry = carry >> 32;
    }
    res_hi[n - 1] = acc;
}

u128b mul65c(uint64_t x1, uint64_t x2, uint64_t y1, uint64_t y2, uint64_t *carry)
{
    uint64_t c = 0;
    uint64_t t1 = x1 - x2;
    uint64_t t2 = y2 - y1;
    u128b result = (u128b) t1 * t2;

    if ((x2 > x1) && t2)
    {
        result -= (u128b) t2 << 64;
        c = ~c;
    }

    if ((y1 > y2) && t1)
    {
        result -= (u128b) t1 << 64;
        c = ~c;
    }

    *carry = c;
    return result;
}

u128b mul65d(uint64_t x1, uint64_t x2, uint64_t y1, uint64_t y2)
{
    u128b result = (u128b) x1 * -y1;

    if ((x2 > x1) && -y1)
    {
        result -= (u128b) - y1 << 64;
    }

    if ((y1 > y2) && x1)
    {
        result -= (u128b) x1 << 64;
    }

    return result;
}

void fast_wide_mul128(uint32_t *res_hi, uint32_t *res_lo, const uint32_t *a, const uint32_t *b)
{
    uint128b a128, b128;

    uint64_t LOW_MASK_64 = 0x00000000FFFFFFFFull;
    u128b LOW_MASK_128 = 0x000000000000000000000000FFFFFFFF;

    a128.lo = uint64_t(a[0]) + (uint64_t(a[1]) << 32);
    a128.mid = uint64_t(a[2]) + (uint64_t(a[3]) << 32);

    b128.lo = uint64_t(b[0]) + (uint64_t(b[1]) << 32);
    b128.mid = uint64_t(b[2]) + (uint64_t(b[3]) << 32);

    uint64_t a1 = a128.lo;
    uint64_t a2 = a128.mid;

    uint64_t b1 = b128.lo;
    uint64_t b2 = b128.mid;

    u128b ab11 = (u128b) a1 * b1;
    u128b ab22 = (u128b) a2 * b2;
    uint64_t ab23 = (0 - a2) * (b2 - 0);

    uint64_t carry;

    u128b t1 = ab11;
    u128b t2 = mul65c(a1, a2, b1, b2, &carry);

    u128b t3 = mul65d(a1, 0, b1, 0) + ab11 + ab22 + (t2 >> 64);
    uint64_t t4 = ab23 + ab22 + carry;

    t2 = (uint64_t) t2;
    t2 += t1 >> 64;
    t3 += t2 >> 64;

    t2 = (uint64_t) t2;
    t2 += ab11;
    t3 += t2 >> 64;

    t2 = (uint64_t) t2;
    t2 += ab22;
    t3 += t2 >> 64;

    res_lo[0] = uint32_t(t1 & LOW_MASK_64);
    res_lo[1] = uint32_t((t1 >> 32) & LOW_MASK_64);
    res_lo[2] = uint32_t(t2 & LOW_MASK_64);
    res_lo[3] = uint32_t((t2 >> 32) & LOW_MASK_64);

    u128b resHi = t3 + ((u128b) t4 << 64);

    res_hi[0] = uint32_t(resHi & LOW_MASK_128);
    res_hi[1] = uint32_t((resHi >> 32) & LOW_MASK_128);
    res_hi[2] = uint32_t((resHi >> 64) & LOW_MASK_128);
    res_hi[3] = uint32_t((resHi >> 96) & LOW_MASK_128);
}

//! Return x as a double, which is obviously lossy for large n
double wide_as_double(unsigned n, const uint32_t *x)
{
    double acc = 0;
    for (unsigned i = 0; i < n; i++)
    {
        acc += ldexp((double)x[i], i * 32);
    }
    return acc;
}

#endif
