/*
   C code to perform matches on the sphere using healpix
*/

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "v2d.h"

//
// code for tests
//

#define dot2(va, vb) ((va).x*(vb).x + (va).y*(vb).y)

#define norm2(v) {					\
	r2d_real tmplen = sqrt(dot2((v), (v)));	\
	(v).x /= (tmplen + 1.0e-299);		\
	(v).y /= (tmplen + 1.0e-299);		\
}

static inline int assert_eq(double x, double y, double tol) {
    int status=0;
    double err = fabs(1.0-x/y);
    if (err > tol) {
        status = 0;
		printf("\x1b[31massert_eq( %.3e , %.3e , %.3e ) failed ( err = %.3e ):\x1b[0m\n  %s, line %d.\n",
				x, y, tol, err, __FILE__, __LINE__);
	} else {
        status=1;
    }

    return status;
}

// warns if two floating-point numbers are not within some fractional tolerance
static inline int expect_eq(double x, double y, double tol)
{
    int status=0;
	double err = fabs(1.0-x/y);
	if(err > tol) {
        status=0;
		printf("\x1b[33mexpect_eq( %.3e , %.3e , %.3e ) failed ( err = %.3e ):\x1b[0m\n  %s, line %d.\n",
               x, y, tol, err, __FILE__, __LINE__);
	} else {
        status=1;
    }
    return status;
}



double rand_uniform(void) {
	// uniform random in (0, 1)
	return ((double) rand())/RAND_MAX;
}

double rand_normal(void) {
	// uses a Box-Muller transform to get two normally distributed numbers
	// from two uniformly distributed ones. We throw one away here.
	double u1 = rand_uniform();
	double u2 = rand_uniform();
	return sqrt(-2.0*log(u1))*cos(6.28318530718*u2);
	//return sqrt(-2.0*log(u1))*sin(TWOPI*u2);
}


r2d_rvec2 rand_uvec_2d(void) {
	// generates a random, isotropically distributed unit vector
	r2d_rvec2 tmp;
	tmp.x = rand_normal();
	tmp.y = rand_normal();
	norm2(tmp);
	return tmp;
}


//
// should be using proper containers for verts
//
r2d_real rand_tri_2d(r2d_rvec2 *verts, int nverts, r2d_real minvol) {
	// generates a random triangle with vertices on the unit circle,
	// guaranteeing a volume of at least minvol (to avoid degenerate cases)

    if (nverts != 3) {
        return 0.0;
    }

	r2d_int v;
	r2d_rvec2 swp;
	r2d_real tetvol = 0.0;
	while(tetvol < minvol) {
		for(v = 0; v < 3; ++v) 
			verts[v] = rand_uvec_2d();
		tetvol = r2d_orient(verts[0], verts[1], verts[2]);
		if(tetvol < 0.0) {
			swp = verts[1];
			verts[1] = verts[2];
			verts[2] = swp;
			tetvol = -tetvol;
		}
	}
	return tetvol;
}


static PyObject *
Pyr2d_test_raster(PyObject* self, PyObject* args)
{

    // Test r2d_rasterize() by checking that the rasterized moments
    // do indeed sum to those of the original input

    int status=0;
    int poly_order=3;
    int ngrid=17;
    double min_area=1.0e-8;
    double tol_warn=1.0e-8;
    double tol_fail=1.0e-4;
    r2d_real* grid = NULL;

    // vars
    r2d_int i, v, curorder, mind;
    r2d_long gg; 
    r2d_int nmom = R2D_NUM_MOMENTS(poly_order);
    r2d_real voxsum, tmom[nmom];
    r2d_poly poly;
    r2d_real area=0.0;

    // note the original test mistakenly declared size 4
    r2d_rvec2 verts[3];

    // create a random tet in the unit box
    area = rand_tri_2d(verts, sizeof(verts), min_area);
    if (area <= 0.0) {
        status=0;
        goto test_raster_bail;
    }
    for(v = 0; v < 3; ++v) {
        for(i = 0; i < 2; ++i) {
            verts[v].xy[i] += 1.0;
            verts[v].xy[i] *= 0.5;
        }
    }
    r2d_init_poly(&poly, verts, 3);

    // get its original moments for reference
    r2d_reduce(&poly, tmom, poly_order);

    // rasterize it
    r2d_rvec2 dx = {{1.0/ngrid, 1.0/ngrid}};
    r2d_dvec2 ibox[2];
    r2d_get_ibox(&poly, ibox, dx);
    printf("Rasterizing a triangle to a grid with dx = %f %f and moments of order %d\n",
           dx.x, dx.y, poly_order);
    printf("Minimum index box = %d %d to %d %d\n", ibox[0].i, ibox[0].j, ibox[1].i, ibox[1].j);
    r2d_int npix = (ibox[1].i-ibox[0].i)*(ibox[1].j-ibox[0].j);
    grid = (r2d_real *) calloc(npix*nmom, sizeof(r2d_real));
    if (!grid) {
        fprintf(stderr,"failed to allocate grid\n");
        goto test_raster_bail;
    }
    r2d_rasterize(&poly, ibox, grid, dx, poly_order);

    // make sure the sum of each moment equals the original 
    for(curorder = 0, mind = 0; curorder <= poly_order; ++curorder) {
        //printf("Order = %d\n", curorder);
        for(i = curorder; i >= 0; --i, ++mind) {
            //j = curorder - i;
            voxsum = 0.0;
            for(gg = 0; gg < npix; ++gg) voxsum += grid[nmom*gg+mind];
            //printf(" Int[ x^%d y^%d dV ] original = %.10e, voxsum = %.10e, error = %.10e\n", 
            //i, j, tmom[mind], voxsum, fabs(1.0 - tmom[mind]/voxsum));
            status=assert_eq(tmom[mind], voxsum, tol_fail);
            if (!status) {
                goto test_raster_bail;
            }
            status=expect_eq(tmom[mind], voxsum, tol_warn);
            if (!status) {
                goto test_raster_bail;
            }
        }
    }

    status=1;
test_raster_bail:
    free(grid);
    return Py_BuildValue("i", status);

}


static PyObject *
Pyr2d_test(PyObject* self, PyObject* args)
{
    int i=0;

    if (!PyArg_ParseTuple(args, (char*)"i", &i)) {
        return NULL;
    }

    return Py_BuildValue("i", i);

}


static PyMethodDef r2d_module_methods[] = {
    {"test",      (PyCFunction)Pyr2d_test, METH_VARARGS,  "a quick test."},
    {"test_raster",      (PyCFunction)Pyr2d_test_raster, METH_NOARGS,  "test raster."},
    {NULL}
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_r2d",      /* m_name */
        "some r2d wrapped methods",  /* m_doc */
        -1,                  /* m_size */
        r2d_module_methods, /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif



#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__r2d(void) 
#else
init_r2d(void) 
#endif
{
    PyObject* m;


#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    m = Py_InitModule3("_r2d", r2d_module_methods, "Define module methods.");
    if (m==NULL) {
        return;
    }
#endif

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
