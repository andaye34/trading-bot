import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <nav style={styles.navbar}>
            <h1 style={styles.logo}>Trading Bot</h1>
            <div style={styles.links}>
                <Link to="/" style={styles.link}>Home</Link>
                <Link to="/analysis" style={styles.link}>Analysis</Link>
            </div>
        </nav>
    );
};

const styles = {
    navbar: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '1rem 2rem',
        backgroundColor: '#222',
        color: '#fff',
    },
    logo: { fontSize: '1.5rem', fontWeight: 'bold' },
    links: { display: 'flex', gap: '1rem' },
    link: { color: '#fff', textDecoration: 'none', fontSize: '1.2rem' },
};

export default Navbar;
