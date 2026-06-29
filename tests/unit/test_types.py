def test_package_imports_with_version():
    import llm_exec_core

    assert llm_exec_core.__version__ == "0.1.0"
