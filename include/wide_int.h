#ifndef wide_int_h
#define wide_int_h

#include <stdint.h>
#include <assert.h>

#include <stdio.h>
#include <math.h>

/*! Simply library for maintaining large positive integers as an array
	of 32-bit limbs */

/*! Compare two integers as numbers:
		a<b :  -1
		a==b :  0
		a>b : +1
*/
int wide_compare(unsigned n, const uint32_t *a, const uint32_t *b)
{
	if(a==b)
		return 0;

	for(int i=n-1;i>=0;i--){
		if(a[i]<b[i])
			return -1;
		if(a[i]>b[i])
			return +1;
	}
	return 0;
}

/*! Copy a source number to a destination */
void wide_copy(unsigned n, uint32_t *res, const uint32_t *a)
{
	for(unsigned i=0;i<n;i++){
		res[i]=a[i];
	}
}

/*! Set entire number to zero */
void wide_zero(unsigned n, uint32_t *res)
{
	for(unsigned i=0;i<n;i++){
		res[i]=0;
	}
}

/*! Set entire number to zero */
void wide_ones(unsigned n, uint32_t *res)
{
	for(unsigned i=0;i<n;i++){
		res[i]=0xFFFFFFFFul;
	}
}

/*! Add together two n-limb numbers, returning the carry limb.
	\note the output can also be one of the inputs
*/
void wide_xor(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
	for(unsigned i=0;i<n;i++){
		res[i]=a[i]^b[i];
	}
}

/*! Add together two n-limb numbers, returning the carry limb.
	\note the output can also be one of the inputs
*/
uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
	uint64_t carry=0;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=uint64_t(a[i])+b[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}


	/*! Add a single limb to an n-limb number, returning the carry limb
	\note the output can also be the input
*/
uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint32_t b)
{
	uint64_t carry=b;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=a[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint64_t b)
{
	assert(n>=2);
	uint64_t acc=a[0]+(b&0xFFFFFFFFULL);
	res[0]=uint32_t(acc&0xFFFFFFFFULL);
	uint64_t carry=acc>>32;
	
	acc=a[1]+(b&0xFFFFFFFFULL)+carry;
	res[1]=uint32_t(acc&0xFFFFFFFFULL);
	carry=acc>>32;
	
	for(unsigned i=2;i<n;i++){
		uint64_t tmp=a[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

/*! Multiply two n-limb numbers to produce a 2n-limb result
	\note All the integers must be distinct, the output cannot overlap the input */
void wide_mul(unsigned n, uint32_t *res_hi, uint32_t *res_lo, const uint32_t *a, const uint32_t *b)
{
	assert(res_hi!=a && res_hi!=b);
	assert(res_lo!=a && res_lo!=b);
	
	uint64_t carry=0, acc=0;
	for(unsigned i=0; i<n; i++){
		for(unsigned j=0; j<=i; j++){
			assert( (j+(i-j))==i );
			uint64_t tmp=uint64_t(a[j])*b[i-j];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,i-j);
		}
		res_lo[i]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i, res_lo[i]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	
	for(unsigned i=1; i<n; i++){
		for(unsigned j=i; j<n; j++){
			uint64_t tmp=uint64_t(a[j])*b[n-j+i-1];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,n-j+i-1);
			//assert( (j+(n-j))==n+i );
		}
		res_hi[i-1]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i+n-1, res_hi[i-1]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	res_hi[n-1]=acc;
}

//! Return x as a double, which is obviously lossy for large n
double wide_as_double(unsigned n, const uint32_t *x)
{
	double acc=0;
	for(unsigned i=0;i<n;i++){
		acc+=ldexp((double)x[i], i*32);
	}
	return acc;
}

#endif
