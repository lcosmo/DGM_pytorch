// This file contains a header automatically completed by cmake with either the user options or default options.
// It define the template F containing a formula to be intentiate. The formula may be defined in two possible ways: 
//          1) with the user friendly "new syntax"  in FORMULA_OBJ variable with possibly aliases in the variable VAR_ALIASES
//          2) with the machine friendly templated syntax in a variable FORMULA  where the operation are template separated by < >

#pragma once

#include <keops_includes.h>

namespace keops {

// specify type for md5 uniqueness: float

#define FORMULA_OBJ_STR "Sum_Reduction(Square((u|v))*Exp(-p*SqNorm2(x-y))*b,0)"
#define VAR_ALIASES_STR "auto x=Vi(1,3); auto y=Vj(2,3); auto u=Vi(3,4); auto v=Vj(4,4); auto b=Vj(5,3); auto p=Pm(0,1);"

static const int NARGS=6;
static const int POS_FIRST_ARGI=1;
static const int POS_FIRST_ARGJ=2;
auto x=Vi(1,3); auto y=Vj(2,3); auto u=Vi(3,4); auto v=Vj(4,4); auto b=Vj(5,3); auto p=Pm(0,1);

#define USENEWSYNTAX 1

#if USENEWSYNTAX

#define FORMULA_OBJ Sum_Reduction(Square((u|v))*Exp(-p*SqNorm2(x-y))*b,0)
using F = decltype(InvKeopsNS(FORMULA_OBJ));

#else

/* #undef FORMULA */
using F = FORMULA;

#endif
}
