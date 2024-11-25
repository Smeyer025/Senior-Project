import MenuElement from "./MenuElement/MenuElement";
import "./MenuContainer.css";

/**
 * MenuContainer()
 * 
 * NAME
 *    MenuContainer() - Handles layout for performance metrics buttons
 * 
 * SYNOPSIS
 *    React.JSX.Element MenuContainer()
 * 
 * DESCRIPTION
 *    This function exports a grid of MenuElements that make up 
 *    the performance metrics menu
 */
export default function MenuContainer() {
    return (
        <>
            <div className="MenuBackground">
                <MenuElement className="Title"><h2>Performance Metrics</h2></MenuElement>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
            </div>

        </>
    );
}