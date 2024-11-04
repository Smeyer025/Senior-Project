/**
 * Header Component: Header of the website
 */

import './Header.css';

/**
 * Header()
 * 
 * NAME
 *    Header() - Makes a header for the website
 * 
 * SYNOPSIS
 *    React.JSX.Element Header(props)
 *      props --> properties
 *          props.children --> Children prop, only default prop.
 *                             Contains the text to be displayed 
 *                             in the Header.
 * 
 * DESCRIPTION
 *    This function exports the Header layout as a component
 *    for it to be used in the App component
 */
export default function Header(props){
    return (
        <header>
            <h1>
                {props.children}
            </h1>
        </header>
    );
}