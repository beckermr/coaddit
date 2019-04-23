/*
   C code to perform matches on the sphere using healpix
*/

#include <Python.h>
#include <numpy/arrayobject.h> 

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
